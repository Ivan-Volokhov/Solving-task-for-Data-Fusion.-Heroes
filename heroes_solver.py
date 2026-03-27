import os
import time
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative


# VISIT_COST — сколько очков хода тратится на посещение мельницы.
VISIT_COST = 100

# HERO_COST — плата за героя
HERO_COST = 2500

# Задача ограничена 7 днями.
MAX_DAY = 7

# Число используемых героев. Можно использовать одно или несколько значений
# На данный момент решается задача для 20 геров
K_LIST = [20]

# очередной кандидат на добавление в маршрут.

# Штраф за старт нового героя.
# Нужен, чтобы алгоритм не размазывал объекты по слишком многим героям.
START_PENALTY = 1700.0

# Штраф за длину переезда.
TRAVEL_W = 1.0

# Штраф за ожидание до дня открытия объекта.
WAIT_W = 250.0

# Штраф за поздний день посещения.
DAY_W = 50.0

# Небольшая награда за остаток очков хода после действия.
REM_W = 0.15

# Случайный шум, чтобы не залипать в не очень глубоком локальном минимуме.
NOISE = 40.0


# Параметры 1-й фазы улучшения.
# Здесь активно улучшаем префиксы до дня 1.
PHASE1_ITERS = 120
PHASE1_SUBSETS = [6, 8, 10, 12, 14, 16, 20]
PHASE1_BEAMS = [100, 140, 180, 220]
PHASE1_EXPANDS = [18, 22, 26, 30]

# Параметры 2-й фазы улучшения.
# Здесь перестраиваем уже префиксы до дня 3.
PHASE2_ITERS = 80
PHASE2_SUBSETS = [4, 5, 6, 7, 8]
PHASE2_BEAMS = [80, 100, 120, 140]
PHASE2_EXPANDS = [14, 18, 22]

# Каждые INSERT_EVERY итераций пробуем дополнительно вставить ещё не посещённые мельницы.
INSERT_EVERY = 20


# Визуализация
VIS_STEPS_PER_DAY = 30
VIS_OUT_HTML = "heroes_visualization.html"

DIRECTORY = r"C://Users//Иван Волохов//.spyder-py3//data//"


# Импорт данных

def read_distance_matrix(path: str) -> np.ndarray:

    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number]).copy()

    if numeric.shape[1] < 700:
        arr = df.to_numpy()
        if arr.shape[1] < 700:
            raise ValueError(f"Не удалось прочитать 700x700 матрицу расстояний из {path}")
        numeric = pd.DataFrame(arr[:, -700:])

    if numeric.shape[1] > 700:
        numeric = numeric.iloc[:, -700:]

    arr = numeric.to_numpy(dtype=np.int32)

    if arr.shape != (700, 700):
        raise ValueError(f"Матрица расстояний должна быть 700x700, получено {arr.shape} из {path}")

    return arr


def read_start_dist(path: str) -> np.ndarray:
    df = pd.read_csv(path)

    if "object_id" not in df.columns:
        id_cols = [c for c in df.columns if "object" in c.lower() and "id" in c.lower()]
        if id_cols:
            df = df.rename(columns={id_cols[0]: "object_id"})

    if "dist_start" not in df.columns:
        dist_cols = [c for c in df.columns if "dist" in c.lower()]
        if dist_cols:
            df = df.rename(columns={dist_cols[-1]: "dist_start"})

    if "object_id" not in df.columns or "dist_start" not in df.columns:
        raise ValueError(f"В {path} ожидаются столбцы object_id и dist_start")

    result = np.zeros(701, dtype=np.int32)

    for _, row in df.iterrows():
        obj = int(row["object_id"])
        if 1 <= obj <= 700:
            result[obj] = int(row["dist_start"])

    return result


# Загружаем таблицы.
heroes = pd.read_csv(os.path.join(DIRECTORY, "data_heroes.csv"))
objects = pd.read_csv(os.path.join(DIRECTORY, "data_objects.csv"))
start_dist = read_start_dist(os.path.join(DIRECTORY, "dist_start.csv"))
D = read_distance_matrix(os.path.join(DIRECTORY, "dist_objects.csv"))


# move_points[hero_id] = запас очков хода героя.
move_points = np.zeros(101, dtype=np.int32)
for _, row in heroes.iterrows():
    move_points[int(row.hero_id)] = int(row.move_points)

# day_open[obj_id] = день, в который объект становится доступен.
day_open = np.zeros(701, dtype=np.int16)
for _, row in objects.iterrows():
    day_open[int(row.object_id)] = int(row.day_open)

# Удобные списки объектов по дням открытия.
ALL_BY_DAY = {
    d: [int(x) for x in objects.loc[objects.day_open == d, "object_id"].tolist()]
    for d in range(1, 8)
}

# Все объекты, доступные в первый день.
ALL_DAY1 = ALL_BY_DAY[1]

# Все объекты, доступные в дни 1..3.
ALL_DAY123 = [obj for d in [1, 2, 3] for obj in ALL_BY_DAY[d]]


# Логика симуляция движения
def advance_state(day: int, rem: int, dist: int, mp: int):

    day = int(day)
    rem = int(rem)
    dist = int(dist)
    mp = int(mp)

    while dist > 0:
        if rem <= 0:
            day += 1
            if day > MAX_DAY:
                return None
            rem = mp
            continue

        step = min(rem, dist)
        rem -= step
        dist -= step

    return day, rem


def eval_append(hero_id: int, state, obj: int):
    mp = int(move_points[hero_id])
    open_day = int(day_open[obj])

    # Случай, когда маршрут героя пуст.
    if state is None:
        travel = int(start_dist[obj])

        # Если объект открывается после горизонта или до него физически не доехать в день старта,
        # то стартовать на нём бессмысленно / нельзя.
        if open_day > MAX_DAY or travel >= mp:
            return None

        # Стартуем в день открытия объекта.
        rem_after_travel = mp - travel

        # После посещения тратим VISIT_COST, если хватает.
        # Если не хватает, остаток считаем 0, и следующий день начнётся с полного mp.
        rem_after_visit = rem_after_travel - VISIT_COST if rem_after_travel >= VISIT_COST else 0
        return open_day, rem_after_visit, obj, travel, 0

    cur_day, cur_rem, cur_loc = state

    # Нельзя "вовремя" посетить объект, который открылся раньше текущего дня героя.
    if open_day < cur_day:
        return None

    travel = int(D[cur_loc - 1, obj - 1])
    adv = advance_state(cur_day, cur_rem, travel, mp)
    if adv is None:
        return None

    arrival_day, rem_after_travel = adv

    # Прибыли раньше — ждём до открытия.
    if arrival_day < open_day:
        rem_after_visit = mp - VISIT_COST
        wait_days = open_day - arrival_day
        return open_day, rem_after_visit, obj, travel, wait_days

    # Прибыли ровно в день открытия и успели приехать не "в ноль".
    if arrival_day == open_day and rem_after_travel >= 1:
        rem_after_visit = rem_after_travel - VISIT_COST if rem_after_travel >= VISIT_COST else 0
        return open_day, rem_after_visit, obj, travel, 0

    # Прибыли позже открытия — как кандидат на "вовремя" уже не подходит.
    return None


def simulate_route(route, hero_id: int):

    mp = int(move_points[hero_id])

    started = False
    day = 0
    rem = 0
    loc = 0

    timely = 0
    per_day = {d: 0 for d in range(1, 8)}

    for obj in route:
        open_day = int(day_open[obj])

        # Первый объект в маршруте: старт из замка.
        if not started:
            travel = int(start_dist[obj])

            if open_day > MAX_DAY or travel >= mp:
                return timely, per_day

            day = open_day
            rem_after_travel = mp - travel

            # ok=True означает, что объект успели посетить вовремя.
            ok = rem_after_travel >= 1

            rem = rem_after_travel - VISIT_COST if rem_after_travel >= VISIT_COST else 0
            loc = obj
            started = True

            if ok:
                timely += 1
                per_day[open_day] += 1

            continue

        travel = int(D[loc - 1, obj - 1])
        adv = advance_state(day, rem, travel, mp)

        if adv is None:
            return timely, per_day

        arrival_day, rem_after_travel = adv

        # Прибыли раньше открытия — ждём и посещаем в день открытия.
        if arrival_day < open_day:
            day = open_day
            rem = mp - VISIT_COST
            ok = True

        # Прибыли ровно в день открытия и не в самый последний тик.
        elif arrival_day == open_day and rem_after_travel >= 1:
            day = arrival_day
            rem = rem_after_travel - VISIT_COST if rem_after_travel >= VISIT_COST else 0
            ok = True

        # Прибыли позже — объект числится посещённым, но не вовремя.
        else:
            day = arrival_day
            rem = rem_after_travel - VISIT_COST if rem_after_travel >= VISIT_COST else 0
            ok = False

        loc = obj

        if ok:
            timely += 1
            per_day[open_day] += 1

    return timely, per_day


def score_routes(routes):
    total_timely = 0
    max_hero = 0
    per_day = {d: 0 for d in range(1, 8)}

    for hero_id, route in routes.items():
        if not route:
            continue

        max_hero = max(max_hero, hero_id)
        t, pdict = simulate_route(route, hero_id)
        total_timely += t

        for d in range(1, 8):
            per_day[d] += pdict[d]

    reward = total_timely * 500
    score = reward - HERO_COST * max_hero
    return score, reward, total_timely, max_hero, per_day


def build_solution(k: int, seed: int):

    rng = random.Random(seed)

    # Маршруты только для первых k героев.
    routes = {hero_id: [] for hero_id in range(1, k + 1)}

    # states[hero_id] хранит текущее состояние героя:
    # None или (day, rem, last_object)
    states = {hero_id: None for hero_id in range(1, k + 1)}

    # Сначала ни один объект не назначен.
    remaining = set(range(1, 701))

    while True:
        best_move = None
        best_score = -10**18

        # Перебираем всех героев и все ещё не назначенные объекты.
        for hero_id in range(1, k + 1):
            state = states[hero_id]

            for obj in remaining:
                info = eval_append(hero_id, state, obj)
                if info is None:
                    continue

                new_day, new_rem, _, travel, wait_days = info

                # Эвристическая оценка кандидата.
                # Чем меньше путь и ожидание, тем лучше.
                # Чем раньше день, тем лучше.
                # Небольшой плюс за запас очков.
                score = 10000.0
                score -= TRAVEL_W * travel
                score -= WAIT_W * wait_days
                score -= DAY_W * (new_day - 1)
                score += REM_W * new_rem

                # Открытие нового героя — дополнительный штраф.
                if state is None:
                    score -= START_PENALTY

                # Немного случайности, чтобы не получать один и тот же greedy-path всегда.
                score += rng.random() * NOISE

                if score > best_score:
                    best_score = score
                    best_move = (hero_id, obj, info)

        # Если больше ничего добавить нельзя — стартовое решение готово.
        if best_move is None:
            break

        hero_id, obj, info = best_move
        routes[hero_id].append(obj)
        states[hero_id] = info[:3]
        remaining.remove(obj)

    return routes


def insert_unvisited(routes, k: int):

    visited = set()
    for route in routes.values():
        visited.update(route)

    unvisited = [obj for obj in range(1, 701) if obj not in visited]
    improved = True

    while improved and unvisited:
        improved = False
        best_choice = None
        best_delta = None

        for obj in list(unvisited):
            obj_day = int(day_open[obj])

            for hero_id in range(1, k + 1):
                route = routes[hero_id]
                old_timely, _ = simulate_route(route, hero_id)

                for pos in range(len(route) + 1):
                    # Поддерживаем порядок по дням открытия.
                    # Это не жёсткая математическая необходимость в общем случае,
                    # но очень полезное ограничение для сокращения поиска.
                    if pos > 0 and int(day_open[route[pos - 1]]) > obj_day:
                        continue
                    if pos < len(route) and obj_day > int(day_open[route[pos]]):
                        continue

                    new_route = route[:pos] + [obj] + route[pos:]
                    new_timely, _ = simulate_route(new_route, hero_id)

                    # Рассматриваем только вставки, которые действительно добавляют 1 своевременное посещение.
                    if new_timely == old_timely + 1:
                        prev_obj = route[pos - 1] if pos > 0 else 0
                        next_obj = route[pos] if pos < len(route) else 0

                        # Считаем прирост длины пути (приблизительный локальный delta).
                        if prev_obj == 0:
                            old_move = start_dist[next_obj] if next_obj != 0 else 0
                            add_1 = start_dist[obj]
                        else:
                            old_move = D[prev_obj - 1, next_obj - 1] if next_obj != 0 else 0
                            add_1 = D[prev_obj - 1, obj - 1]

                        add_2 = D[obj - 1, next_obj - 1] if next_obj != 0 else 0
                        delta = int(add_1) + int(add_2) - int(old_move)

                        if best_delta is None or delta < best_delta:
                            best_delta = delta
                            best_choice = (hero_id, pos, obj)

        if best_choice is not None:
            hero_id, pos, obj = best_choice
            routes[hero_id] = routes[hero_id][:pos] + [obj] + routes[hero_id][pos:]
            unvisited.remove(obj)
            improved = True

    return routes


def split_prefix_suffix(routes, day_limit: int):

    prefixes = {}
    suffixes = {}

    for hero_id, route in routes.items():
        i = 0
        while i < len(route) and int(day_open[route[i]]) <= day_limit:
            i += 1

        prefixes[hero_id] = route[:i]
        suffixes[hero_id] = tuple(route[i:])

    return prefixes, suffixes


def make_suffix_checker(hero_id: int, suffix):

    @lru_cache(maxsize=None)
    def can_finish(day: int, rem: int, loc: int):
        state = None if loc == 0 else (day, rem, loc)
        cur = state

        for obj in suffix:
            info = eval_append(hero_id, cur, obj)
            if info is None:
                return False
            cur = info[:3]

        return True

    return can_finish


def build_best_prefix(hero_id: int, available, suffix, day_limit: int, rng, beam_width: int, max_expand: int):

    can_finish = make_suffix_checker(hero_id, suffix)
    target = suffix[0] if suffix else None

    # Элемент beam:
    # (route_prefix, state, heuristic_score, total_cnt, cnt_day)
    beam = [([], None, 0.0, 0, {1: 0, 2: 0, 3: 0})]

    best_route = []
    best_key = (-10**18, -10**18, -10**18, -10**18)

    for _ in range(max_expand):
        cand = []

        for route, state, score, total_cnt, cnt_day in beam:
            # Даже если больше ничего не добавляем, проверим:
            # можно ли уже сейчас корректно достроить suffix.
            if state is None:
                ok = can_finish(0, 0, 0)
            else:
                ok = can_finish(int(state[0]), int(state[1]), int(state[2]))

            if ok:
                key = (total_cnt, cnt_day[1], cnt_day[2] + cnt_day[3], score)
                if key > best_key:
                    best_key = key
                    best_route = list(route)

            used = set(route)

            # Пробуем добавить в префикс ещё один объект.
            for obj in available:
                if obj in used:
                    continue
                if int(day_open[obj]) > day_limit:
                    continue

                info = eval_append(hero_id, state, obj)
                if info is None:
                    continue

                st = info[:3]

                # Префикс должен оставаться внутри day_limit.
                if int(st[0]) > day_limit:
                    continue

                # После добавления объекта хвост должен всё ещё быть достижим.
                if not can_finish(int(st[0]), int(st[1]), int(st[2])):
                    continue

                # Локальные признаки качества расширения.
                if state is None:
                    travel = int(start_dist[obj])
                    cur_loc = 0
                else:
                    travel = int(D[state[2] - 1, obj - 1])
                    cur_loc = int(state[2])

                # Если есть target = первый объект suffix,
                # полезно не только собирать объекты, но и продвигаться к этой цели.
                if target is not None:
                    if cur_loc == 0:
                        cur_to_target = int(start_dist[target])
                    else:
                        cur_to_target = int(D[cur_loc - 1, target - 1])

                    to_target = int(D[obj - 1, target - 1])
                    progress = cur_to_target - to_target
                else:
                    to_target = 0
                    progress = 0

                d = int(day_open[obj])
                new_cnt = {1: cnt_day[1], 2: cnt_day[2], 3: cnt_day[3]}
                if d <= 3:
                    new_cnt[d] += 1

                # Эвристическая оценка для beam search.
                new_score = score
                new_score += 10000.0              # базовая награда за ещё один объект
                new_score -= float(travel)        # короткие переезды лучше
                new_score -= 0.8 * float(to_target)
                new_score += 2.6 * float(progress)
                new_score += 0.15 * float(st[1])  # запас хода полезен
                new_score -= 60.0 * float(max(0, int(st[0]) - 1))  # ранние дни лучше

                # Дополнительный приоритет более ранним дням.
                if d == 1:
                    new_score += 1800.0
                elif d == 2:
                    new_score += 500.0
                elif d == 3:
                    new_score += 250.0

                new_score += rng.random() * 15.0
                cand.append((route + [obj], st, new_score, total_cnt + 1, new_cnt))

        if not cand:
            break

        # Оставляем самые интересные частичные маршруты.
        cand.sort(key=lambda x: (x[3], x[4][1], x[4][2] + x[4][3], x[2]), reverse=True)

        new_beam = []
        seen = set()

        for route, state, score, total_cnt, cnt_day in cand:
            # Упрощённый сигнатурный ключ, чтобы не держать слишком похожие состояния.
            key = (int(state[0]), int(state[2]), int(state[1]) // 25, total_cnt, cnt_day[1])

            if key in seen:
                continue

            seen.add(key)
            new_beam.append((route, state, score, total_cnt, cnt_day))

            if len(new_beam) >= beam_width:
                break

        beam = new_beam

    # Финальная проверка состояний из beam.
    for route, state, score, total_cnt, cnt_day in beam:
        if state is None:
            ok = can_finish(0, 0, 0)
        else:
            ok = can_finish(int(state[0]), int(state[1]), int(state[2]))

        if ok:
            key = (total_cnt, cnt_day[1], cnt_day[2] + cnt_day[3], score)
            if key > best_key:
                best_key = key
                best_route = list(route)

    return best_route


def rebuild_subset_prefix(routes, subset, k: int, day_limit: int, rng, beam_width: int, max_expand: int):

    prefixes, suffixes = split_prefix_suffix(routes, day_limit)

    other_heroes = [h for h in range(1, k + 1) if h not in subset]
    locked = set()

    # Всё, что уже занято "не трогаемыми" героями, недоступно для перестройки.
    for hero_id in other_heroes:
        locked.update(prefixes[hero_id])

    pool = ALL_DAY1 if day_limit == 1 else ALL_DAY123
    available = set(obj for obj in pool if obj not in locked)
    new_routes = {hero_id: list(routes[hero_id]) for hero_id in other_heroes}

    # Порядок перестройки героев тоже влияет на результат.
    order = subset[:]
    mode = rng.choice(["fewfirst", "move_desc", "target_far", "random"])

    if mode == "fewfirst":
        order.sort(key=lambda h: (len(prefixes[h]), -int(move_points[h])))
    elif mode == "move_desc":
        order.sort(key=lambda h: int(move_points[h]), reverse=True)
    elif mode == "target_far":
        order.sort(
            key=lambda h: int(start_dist[suffixes[h][0]]) if len(suffixes[h]) > 0 else 0,
            reverse=True,
        )
    else:
        rng.shuffle(order)

    for hero_id in order:
        prefix = build_best_prefix(
            hero_id=hero_id,
            available=available,
            suffix=suffixes[hero_id],
            day_limit=day_limit,
            rng=rng,
            beam_width=beam_width,
            max_expand=max_expand,
        )

        new_routes[hero_id] = prefix + list(suffixes[hero_id])

        # Объекты, использованные этим героем в новом префиксе, убираем из пула.
        for obj in prefix:
            available.discard(obj)

    for hero_id in range(1, k + 1):
        if hero_id not in new_routes:
            new_routes[hero_id] = list(routes[hero_id])

    return new_routes


def routes_to_df(routes):
    rows = []
    seen_global = set()

    for hero_id in sorted(routes):
        seen_local = set()

        for obj in routes[hero_id]:
            if obj not in seen_local and obj not in seen_global:
                rows.append((hero_id, obj))
                seen_local.add(obj)
                seen_global.add(obj)

    return pd.DataFrame(rows, columns=["hero_id", "object_id"])


def find_xy_columns(df):

    pairs = [
        ("x", "y"),
        ("coord_x", "coord_y"),
        ("pos_x", "pos_y"),
        ("X", "Y"),
        ("lon", "lat"),
        ("lng", "lat"),
        ("longitude", "latitude"),
    ]

    cols = set(df.columns)

    for cx, cy in pairs:
        if cx in cols and cy in cols:
            return cx, cy

    return None, None


def classical_mds(dist_matrix: np.ndarray):

    n = dist_matrix.shape[0]
    d2 = dist_matrix.astype(np.float64) ** 2
    j = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    b = -0.5 * j @ d2 @ j

    eigvals, eigvecs = np.linalg.eigh(b)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    pos = eigvals > 1e-9
    eigvals = eigvals[pos][:2]
    eigvecs = eigvecs[:, pos][:, :2]

    if eigvals.size == 0:
        return np.zeros((n, 2), dtype=np.float64)

    x = eigvecs * np.sqrt(eigvals)

    if x.shape[1] < 2:
        x = np.hstack([x, np.zeros((n, 2 - x.shape[1]), dtype=np.float64)])

    return x


def get_visual_coordinates():

    cx, cy = find_xy_columns(objects)

    x_by_obj = np.zeros(701, dtype=np.float64)
    y_by_obj = np.zeros(701, dtype=np.float64)

    if cx is not None and cy is not None:
        for _, row in objects.iterrows():
            obj = int(row.object_id)
            x_by_obj[obj] = float(row[cx])
            y_by_obj[obj] = float(row[cy])

        sx = float(np.mean([x_by_obj[obj] for obj in range(1, 701)]))
        sy = float(np.mean([y_by_obj[obj] for obj in range(1, 701)]))
        return x_by_obj, y_by_obj, (sx, sy)

    # Добавляем "узел 0" как замок и строим общую матрицу расстояний 701 x 701.
    full = np.zeros((701, 701), dtype=np.float64)

    for obj in range(1, 701):
        full[0, obj] = float(start_dist[obj])
        full[obj, 0] = float(start_dist[obj])

    full[1:, 1:] = D.astype(np.float64)
    coords = classical_mds(full)

    for obj in range(1, 701):
        x_by_obj[obj] = float(coords[obj, 0])
        y_by_obj[obj] = float(coords[obj, 1])

    return x_by_obj, y_by_obj, (float(coords[0, 0]), float(coords[0, 1]))


def interp(x1: float, y1: float, x2: float, y2: float, t: float):

    return x1 + (x2 - x1) * t, y1 + (y2 - y1) * t



def simulate_route_detailed(route, hero_id: int, x_by_obj, y_by_obj, start_xy):

    mp = int(move_points[hero_id])

    segments_by_day = {d: [] for d in range(1, 8)}
    day_start_pos = {}
    visits = []

    if not route:
        return segments_by_day, day_start_pos, visits

    started = False
    day = 0
    rem = 0
    loc = 0
    current_x, current_y = float(start_xy[0]), float(start_xy[1])

    for idx, obj in enumerate(route, start=1):
        obj_x = float(x_by_obj[obj])
        obj_y = float(y_by_obj[obj])
        open_day = int(day_open[obj])

        # Первый объект — движение из замка.
        if not started:
            travel = int(start_dist[obj])

            if open_day > MAX_DAY or travel >= mp:
                break

            day = open_day
            day_start_pos[day] = (float(start_xy[0]), float(start_xy[1]))

            if travel > 0:
                segments_by_day[day].append(
                    {
                        "move_start": 0.0,
                        "move_end": float(travel),
                        "x1": float(start_xy[0]),
                        "y1": float(start_xy[1]),
                        "x2": obj_x,
                        "y2": obj_y,
                    }
                )

            rem_after_travel = mp - travel
            visit_progress = float(travel) / float(mp)
            timely = rem_after_travel >= 1
            rem = rem_after_travel - VISIT_COST if rem_after_travel >= VISIT_COST else 0
            loc = obj
            started = True
            current_x, current_y = obj_x, obj_y

            visits.append(
                {
                    "hero_id": hero_id,
                    "object_id": obj,
                    "route_index": idx,
                    "visit_day": day,
                    "visit_progress": visit_progress,
                    "arrival_day": day,
                    "timely": bool(timely),
                    "timely_day": open_day if timely else None,
                    "timely_progress": visit_progress if timely else None,
                }
            )
            continue

        # Следующий объект: детально раскладываем путь по дням.
        travel = int(D[loc - 1, obj - 1])
        dist_left = travel
        leg_x1, leg_y1 = current_x, current_y
        leg_x2, leg_y2 = obj_x, obj_y
        total_dist = max(1.0, float(travel))

        while dist_left > 0:
            if rem <= 0:
                day += 1
                if day > MAX_DAY:
                    return segments_by_day, day_start_pos, visits
                rem = mp
                day_start_pos[day] = (current_x, current_y)
                continue

            step = min(rem, dist_left)
            p0 = (travel - dist_left) / total_dist
            p1 = (travel - dist_left + step) / total_dist

            sx, sy = interp(leg_x1, leg_y1, leg_x2, leg_y2, p0)
            ex, ey = interp(leg_x1, leg_y1, leg_x2, leg_y2, p1)

            if day not in day_start_pos:
                day_start_pos[day] = (sx, sy)

            move_start = float(mp - rem)
            move_end = float(move_start + step)

            segments_by_day[day].append(
                {
                    "move_start": move_start,
                    "move_end": move_end,
                    "x1": sx,
                    "y1": sy,
                    "x2": ex,
                    "y2": ey,
                }
            )

            rem -= step
            dist_left -= step
            current_x, current_y = ex, ey

        arrival_day = day
        arrival_progress = float(mp - rem) / float(mp)

        if arrival_day < open_day:
            day = open_day
            if day > MAX_DAY:
                return segments_by_day, day_start_pos, visits

            day_start_pos[day] = (obj_x, obj_y)
            visit_day = open_day
            visit_progress = 0.0
            timely = True
            rem = mp - VISIT_COST
        elif arrival_day == open_day and rem >= 1:
            visit_day = arrival_day
            visit_progress = arrival_progress
            timely = True
            rem = rem - VISIT_COST if rem >= VISIT_COST else 0
        else:
            visit_day = arrival_day
            visit_progress = arrival_progress
            timely = False
            rem = rem - VISIT_COST if rem >= VISIT_COST else 0

        loc = obj

        visits.append(
            {
                "hero_id": hero_id,
                "object_id": obj,
                "route_index": idx,
                "visit_day": visit_day,
                "visit_progress": visit_progress,
                "arrival_day": arrival_day,
                "timely": bool(timely),
                "timely_day": open_day if timely else None,
                "timely_progress": visit_progress if timely else None,
            }
        )

    return segments_by_day, day_start_pos, visits


def build_detailed_solution(routes, k: int, x_by_obj, y_by_obj, start_xy):

    detail = {"segments": {}, "day_start_pos": {}, "visits": {}}

    for hero_id in range(1, k + 1):
        segs, starts, visits = simulate_route_detailed(
            routes[hero_id],
            hero_id,
            x_by_obj,
            y_by_obj,
            start_xy,
        )
        detail["segments"][hero_id] = segs
        detail["day_start_pos"][hero_id] = starts
        detail["visits"][hero_id] = visits

    return detail


def event_happened(day: int, alpha: float, event_day, event_progress):

    if event_day is None:
        return False
    if event_day < day:
        return True
    if event_day > day:
        return False
    return float(event_progress) <= float(alpha) + 1e-12


def build_visual_stats(detail, k: int):

    final_timely_by_obj = {}
    final_visit_by_obj = {}
    visits_by_hero = {h: list(detail["visits"][h]) for h in range(1, k + 1)}

    for hero_id in range(1, k + 1):
        for v in detail["visits"][hero_id]:
            obj = int(v["object_id"])
            vd = int(v["visit_day"])
            vp = float(v["visit_progress"])

            cur = final_visit_by_obj.get(obj)
            if cur is None or (vd, vp) < cur:
                final_visit_by_obj[obj] = (vd, vp)

            if v["timely"]:
                td = int(v["timely_day"])
                tp = float(v["timely_progress"])
                curt = final_timely_by_obj.get(obj)

                if curt is None or (td, tp) < curt:
                    final_timely_by_obj[obj] = (td, tp)

    timely_final_set_by_day = {}
    missed_final_set_by_day = {}

    for day in range(1, 8):
        timely_set = {
            obj
            for obj in ALL_BY_DAY[day]
            if obj in final_timely_by_obj and final_timely_by_obj[obj][0] == day
        }

        timely_final_set_by_day[day] = timely_set
        missed_final_set_by_day[day] = set(ALL_BY_DAY[day]) - timely_set

    return {
        "visits_by_hero": visits_by_hero,
        "final_visit_by_obj": final_visit_by_obj,
        "final_timely_by_obj": final_timely_by_obj,
        "timely_final_set_by_day": timely_final_set_by_day,
        "missed_final_set_by_day": missed_final_set_by_day,
    }


def compute_dynamic_counts(visual_stats, day: int, alpha: float, k: int):

    visited_objs = set()
    timely_objs = set()
    hero_visited = {h: 0 for h in range(1, k + 1)}
    hero_timely = {h: 0 for h in range(1, k + 1)}

    for obj, (vd, vp) in visual_stats["final_visit_by_obj"].items():
        if event_happened(day, alpha, vd, vp):
            visited_objs.add(obj)

    for obj, (td, tp) in visual_stats["final_timely_by_obj"].items():
        if event_happened(day, alpha, td, tp):
            timely_objs.add(obj)

    for hero_id in range(1, k + 1):
        for v in visual_stats["visits_by_hero"][hero_id]:
            if event_happened(day, alpha, int(v["visit_day"]), float(v["visit_progress"])):
                hero_visited[hero_id] += 1

            if v["timely"] and event_happened(
                day,
                alpha,
                int(v["timely_day"]),
                float(v["timely_progress"]),
            ):
                hero_timely[hero_id] += 1

    return visited_objs, timely_objs, hero_visited, hero_timely


def current_day_status(day: int, alpha: float, visual_stats):

    green = set()

    for obj in ALL_BY_DAY[day]:
        if obj in visual_stats["final_timely_by_obj"]:
            td, tp = visual_stats["final_timely_by_obj"][obj]
            if td == day and event_happened(day, alpha, td, tp):
                green.add(obj)

    if alpha >= 1.0 - 1e-12:
        red = set(visual_stats["missed_final_set_by_day"][day])
        gold = set(ALL_BY_DAY[day]) - green - red
    else:
        red = set()
        gold = set(ALL_BY_DAY[day]) - green

    return gold, green, red


def hero_path_until(hero_id: int, day: int, alpha: float, detail):

    mp = int(move_points[hero_id])
    move_cut = alpha * mp

    starts = detail["day_start_pos"][hero_id]
    segs = detail["segments"][hero_id][day]

    if day not in starts:
        return [], [], None

    start_pos = starts[day]
    xs = [float(start_pos[0])]
    ys = [float(start_pos[1])]
    cur_pos = (float(start_pos[0]), float(start_pos[1]))

    for s in segs:
        ms = float(s["move_start"])
        me = float(s["move_end"])

        if move_cut <= ms:
            break

        if move_cut >= me:
            xs.append(float(s["x2"]))
            ys.append(float(s["y2"]))
            cur_pos = (float(s["x2"]), float(s["y2"]))
        else:
            frac = (move_cut - ms) / max(1e-9, me - ms)
            cx = float(s["x1"]) + frac * (float(s["x2"]) - float(s["x1"]))
            cy = float(s["y1"]) + frac * (float(s["y2"]) - float(s["y1"]))
            xs.append(cx)
            ys.append(cy)
            cur_pos = (cx, cy)
            break

    return xs, ys, cur_pos


def stats_html(day: int, alpha: float, score: int, visual_stats, k: int):

    visited_objs, timely_objs, hero_visited, hero_timely = compute_dynamic_counts(
        visual_stats,
        day,
        alpha,
        k,
    )

    lines = [
        f"<b>День:</b> {day}",
        f"<b>Прогресс дня:</b> {int(round(alpha * 100))}%",
        "",
        f"<b>Score:</b> {score}",
        f"<b>Всего мельниц:</b> 700",
        f"<b>Посещено героями:</b> {len(visited_objs)}",
        f"<b>Посещено вовремя:</b> {len(timely_objs)}",
        f"<b>Не посещено:</b> {700 - len(visited_objs)}",
        "",
        "<b>Статистика по героям</b>",
    ]

    for hero_id in range(1, k + 1):
        mp = int(move_points[hero_id])
        visited_cnt = int(hero_visited[hero_id])
        timely_cnt = int(hero_timely[hero_id])
        money = timely_cnt * 500
        lines.append(
            f"H{hero_id:02d}: ОХ={mp}, мельниц={visited_cnt}, вовремя={timely_cnt}, золото={money}"
        )

    return "<br>".join(lines)


def create_interactive_visualization(
    routes,
    k: int,
    score: int,
    reward: int,
    timely: int,
    max_hero: int,
    days,
    out_html: str = VIS_OUT_HTML,
):
    x_by_obj, y_by_obj, start_xy = get_visual_coordinates()
    detail = build_detailed_solution(routes, k, x_by_obj, y_by_obj, start_xy)
    visual_stats = build_visual_stats(detail, k)

    all_x = [float(x_by_obj[obj]) for obj in range(1, 701)]
    all_y = [float(y_by_obj[obj]) for obj in range(1, 701)]
    all_text = [f"Мельница {obj}<br>Открытие: день {int(day_open[obj])}" for obj in range(1, 701)]

    colors = qualitative.Dark24 + qualitative.Light24 + qualitative.Alphabet

    fig = go.Figure()

    # База: все мельницы на фоне.
    fig.add_trace(
        go.Scatter(
            x=all_x,
            y=all_y,
            mode="markers",
            marker=dict(size=6, color="lightgray"),
            text=all_text,
            hovertemplate="%{text}<extra></extra>",
            name="Все мельницы",
        )
    )

    # Мельницы, относящиеся к текущему дню.
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(size=10, color="gold"),
            text=[],
            hovertemplate="%{text}<extra></extra>",
            name="Мельницы текущего дня",
        )
    )

    # Уже посещённые вовремя.
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(size=11, color="green"),
            text=[],
            hovertemplate="%{text}<extra></extra>",
            name="Посещены вовремя",
        )
    )

    # Не закрытые вовремя к концу дня.
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker=dict(size=11, color="red"),
            text=[],
            hovertemplate="%{text}<extra></extra>",
            name="Не успели до конца дня",
        )
    )

    # Замок.
    fig.add_trace(
        go.Scatter(
            x=[float(start_xy[0])],
            y=[float(start_xy[1])],
            mode="markers+text",
            marker=dict(size=14, color="black", symbol="x"),
            text=["Замок"],
            textposition="bottom center",
            hovertemplate="Замок<extra></extra>",
            name="Замок",
        )
    )

    # Заготовки трасс героев.
    for hero_id in range(1, k + 1):
        color = colors[(hero_id - 1) % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(width=3, color=color),
                hoverinfo="skip",
                name=f"Путь H{hero_id}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers+text",
                marker=dict(size=12, color=color),
                text=[],
                textposition="top center",
                hovertemplate="Герой %{text}<extra></extra>",
                name=f"H{hero_id}",
            )
        )

    frames = []
    slider_steps = []

    # Строим кадры анимации: каждый день делим на VIS_STEPS_PER_DAY шагов.
    for day in range(1, 8):
        for step in range(VIS_STEPS_PER_DAY):
            alpha = step / max(1, VIS_STEPS_PER_DAY - 1)
            gold_objs, green_objs, red_objs = current_day_status(day, alpha, visual_stats)

            gold_x = [float(x_by_obj[obj]) for obj in sorted(gold_objs)]
            gold_y = [float(y_by_obj[obj]) for obj in sorted(gold_objs)]
            gold_text = [f"Мельница {obj}<br>Открытие: день {day}" for obj in sorted(gold_objs)]

            green_x = [float(x_by_obj[obj]) for obj in sorted(green_objs)]
            green_y = [float(y_by_obj[obj]) for obj in sorted(green_objs)]
            green_text = [
                f"Мельница {obj}<br>Посещена вовремя в день {day}"
                for obj in sorted(green_objs)
            ]

            red_x = [float(x_by_obj[obj]) for obj in sorted(red_objs)]
            red_y = [float(y_by_obj[obj]) for obj in sorted(red_objs)]
            red_text = [
                f"Мельница {obj}<br>Не посещена вовремя к концу дня {day}"
                for obj in sorted(red_objs)
            ]

            data = [
                go.Scatter(
                    x=all_x,
                    y=all_y,
                    mode="markers",
                    marker=dict(size=6, color="lightgray"),
                    text=all_text,
                    hovertemplate="%{text}<extra></extra>",
                    name="Все мельницы",
                ),
                go.Scatter(
                    x=gold_x,
                    y=gold_y,
                    mode="markers",
                    marker=dict(size=10, color="gold"),
                    text=gold_text,
                    hovertemplate="%{text}<extra></extra>",
                    name="Мельницы текущего дня",
                ),
                go.Scatter(
                    x=green_x,
                    y=green_y,
                    mode="markers",
                    marker=dict(size=11, color="green"),
                    text=green_text,
                    hovertemplate="%{text}<extra></extra>",
                    name="Посещены вовремя",
                ),
                go.Scatter(
                    x=red_x,
                    y=red_y,
                    mode="markers",
                    marker=dict(size=11, color="red"),
                    text=red_text,
                    hovertemplate="%{text}<extra></extra>",
                    name="Не успели до конца дня",
                ),
                go.Scatter(
                    x=[float(start_xy[0])],
                    y=[float(start_xy[1])],
                    mode="markers+text",
                    marker=dict(size=14, color="black", symbol="x"),
                    text=["Замок"],
                    textposition="bottom center",
                    hovertemplate="Замок<extra></extra>",
                    name="Замок",
                ),
            ]

            for hero_id in range(1, k + 1):
                color = colors[(hero_id - 1) % len(colors)]
                xs, ys, pos = hero_path_until(hero_id, day, alpha, detail)

                if pos is None:
                    data.append(
                        go.Scatter(
                            x=[],
                            y=[],
                            mode="lines",
                            line=dict(width=3, color=color),
                            hoverinfo="skip",
                            name=f"Путь H{hero_id}",
                        )
                    )
                    data.append(
                        go.Scatter(
                            x=[],
                            y=[],
                            mode="markers+text",
                            marker=dict(size=12, color=color),
                            text=[],
                            textposition="top center",
                            hovertemplate="Герой %{text}<extra></extra>",
                            name=f"H{hero_id}",
                        )
                    )
                else:
                    data.append(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            line=dict(width=3, color=color),
                            hoverinfo="skip",
                            name=f"Путь H{hero_id}",
                        )
                    )
                    data.append(
                        go.Scatter(
                            x=[pos[0]],
                            y=[pos[1]],
                            mode="markers+text",
                            marker=dict(size=12, color=color),
                            text=[f"H{hero_id}"],
                            textposition="top center",
                            hovertemplate="Герой %{text}<extra></extra>",
                            name=f"H{hero_id}",
                        )
                    )

            frames.append(
                go.Frame(
                    name=f"d{day}_s{step}",
                    data=data,
                    layout=go.Layout(
                        annotations=[
                            dict(
                                x=0.74,
                                y=0.98,
                                xref="paper",
                                yref="paper",
                                xanchor="left",
                                yanchor="top",
                                align="left",
                                showarrow=False,
                                text=stats_html(day, alpha, score, visual_stats, k),
                                font=dict(size=11),
                            )
                        ]
                    ),
                )
            )

            slider_steps.append(
                {
                    "label": f"D{day}:{step + 1}",
                    "method": "animate",
                    "args": [
                        [f"d{day}_s{step}"],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                }
            )

    fig.frames = frames

    if frames:
        fig = go.Figure(data=frames[0].data, frames=frames)

    fig.update_layout(
        title="Маршруты героев по дням",
        width=1800,
        height=950,
        showlegend=True,
        hovermode="closest",
        margin=dict(l=40, r=40, t=70, b=40),
        xaxis=dict(domain=[0.0, 0.70], title="X", showgrid=True, zeroline=False),
        yaxis=dict(
            domain=[0.0, 1.0],
            title="Y",
            showgrid=True,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        annotations=[
            dict(
                x=0.74,
                y=0.98,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                align="left",
                showarrow=False,
                text=stats_html(1, 0.0, score, visual_stats, k),
                font=dict(size=11),
            )
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 1.08,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 250, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "x": 0.02,
                "y": -0.02,
                "len": 0.96,
                "currentvalue": {"prefix": "Кадр: "},
                "steps": slider_steps,
            }
        ],
    )

    fig.write_html(out_html, include_plotlyjs="cdn")

def run_for_k(k: int):

    print("\n" + "=" * 80)
    print("RUN FOR K =", k)
    print("=" * 80)

    best_routes = build_solution(k, 0)
    best_routes = insert_unvisited(best_routes, k)

    best_score, best_reward, best_timely, best_max_hero, best_days = score_routes(best_routes)
    print("START:", best_score, best_reward, best_timely, best_max_hero, best_days)

    for it in range(PHASE1_ITERS):
        rng = random.Random(100000 + k * 1000 + it)
        prefixes, _ = split_prefix_suffix(best_routes, 1)
        prefix_sizes = {h: len(prefixes[h]) for h in range(1, k + 1)}

        subset_size = rng.choice([x for x in PHASE1_SUBSETS if x <= k])
        subset = []

        # С повышенным шансом выбираем тех героев, у кого префикс пока короткий.
        while len(subset) < subset_size:
            choices = [h for h in range(1, k + 1) if h not in subset]
            weights = [14 - min(prefix_sizes[h], 12) for h in choices]
            subset.append(rng.choices(choices, weights=weights, k=1)[0])

        cand_routes = rebuild_subset_prefix(
            routes=best_routes,
            subset=subset,
            k=k,
            day_limit=1,
            rng=rng,
            beam_width=rng.choice(PHASE1_BEAMS),
            max_expand=rng.choice(PHASE1_EXPANDS),
        )

        if it % INSERT_EVERY == 0:
            cand_routes = insert_unvisited(cand_routes, k)

        cand_score, cand_reward, cand_timely, cand_max_hero, cand_days = score_routes(cand_routes)

        # Принимаем улучшение, если вырос score.
        # При равном score предпочитаем больше объектов в день 1.
        if cand_score > best_score or (cand_score == best_score and cand_days[1] > best_days[1]):
            best_routes = {h: list(r) for h, r in cand_routes.items()}
            best_score, best_reward, best_timely, best_max_hero, best_days = (
                cand_score,
                cand_reward,
                cand_timely,
                cand_max_hero,
                cand_days,
            )
            print(
                f"P1 improved it={it:3d}  score={best_score:7d}  timely={best_timely:3d}  days={best_days}"
            )


    for it in range(PHASE2_ITERS):
        rng = random.Random(200000 + k * 1000 + it)
        subset_size = rng.choice([x for x in PHASE2_SUBSETS if x <= k])
        subset = rng.sample(list(range(1, k + 1)), subset_size)

        cand_routes = rebuild_subset_prefix(
            routes=best_routes,
            subset=subset,
            k=k,
            day_limit=3,
            rng=rng,
            beam_width=rng.choice(PHASE2_BEAMS),
            max_expand=rng.choice(PHASE2_EXPANDS),
        )

        if it % INSERT_EVERY == 0:
            cand_routes = insert_unvisited(cand_routes, k)

        cand_score, cand_reward, cand_timely, cand_max_hero, cand_days = score_routes(cand_routes)

        if cand_score > best_score or (cand_score == best_score and cand_days[1] > best_days[1]):
            best_routes = {h: list(r) for h, r in cand_routes.items()}
            best_score, best_reward, best_timely, best_max_hero, best_days = (
                cand_score,
                cand_reward,
                cand_timely,
                cand_max_hero,
                cand_days,
            )
            print(
                f"P2 improved it={it:3d}  score={best_score:7d}  timely={best_timely:3d}  days={best_days}"
            )

    return best_routes, best_score, best_reward, best_timely, best_max_hero, best_days

def main():

    t0 = time.time()

    global_best = None
    global_best_score = -10**18
    global_info = None

    for k in K_LIST:
        routes, score, reward, timely, max_hero, days = run_for_k(k)

        if score > global_best_score:
            global_best = routes
            global_best_score = score
            global_info = (score, reward, timely, max_hero, days, k)

    score, reward, timely, max_hero, days, k = global_info

    print("\n" + "#" * 80)
    print("GLOBAL BEST")
    print("score   =", score)
    print("reward  =", reward)
    print("timely  =", timely)
    print("maxhero =", max_hero)
    print("days    =", days)
    print("k       =", k)
    print("elapsed =", round(time.time() - t0, 2), "sec")
    print("#" * 80)

    out = routes_to_df(global_best)
    out.to_csv("solution_heroes.csv", index=False)
    print("saved: solution_heroes.csv")

    create_interactive_visualization(
        routes=global_best,
        k=k,
        score=score,
        reward=reward,
        timely=timely,
        max_hero=max_hero,
        days=days,
        out_html=VIS_OUT_HTML,
    )
    print(f"saved: {VIS_OUT_HTML}")


if __name__ == "__main__":
    main()
