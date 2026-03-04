#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Построение рисунков (RU) для статьи о минтайе северной части Охотского моря.

Запуск:
    python scripts/02_analysis_and_figures_ru.py

Вход:
    data/pollock_okhotsk_climate_fishery_1963_2025.csv

Выход:
    outputs/figures/ (PNG, 300 dpi)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "pollock_okhotsk_climate_fishery_1963_2025.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

plt.rcParams["font.family"] = "DejaVu Sans"

def savefig(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)

# Рис. 1 — биомасса
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["year"], df["TSB_million_t_updated"], label="TSB (общая биомасса)")
ax.plot(df["year"], df["SSB_million_t_updated"], label="SSB (нерестовая биомасса)")
for yr in [1983, 1998, 2008]:
    ax.axvline(yr, linestyle="--", linewidth=1)
ax.set_xlabel("Год")
ax.set_ylabel("млн т")
ax.set_title("Минтай северной части Охотского моря: TSB и SSB (1963–2025)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
savefig(fig, "рис1_биомасса_TSB_SSB_1963_2025.png")

# Рис. 2 — вылов и ОДУ
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["year"], df["catch_total_thousand_t"], label="Годовой вылов (тыс. т)")
ax.scatter(df["year"], df["ODU_total_thousand_t"], label="ОДУ / TAC (тыс. т)")
ax.set_xlabel("Год")
ax.set_ylabel("тыс. т")
ax.set_title("Минтай северной части Охотского моря: вылов и ОДУ (1963–2025)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
savefig(fig, "рис2_вылов_оду_1963_2025.png")

# Рис. 3 — SST и лёд
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(df["year"], df["SST_anom_degC"])
for yr in [1981, 1999, 2007]:
    axes[0].axvline(yr, linestyle="--", linewidth=1)
axes[0].set_ylabel("Аномалия SST, °C")
axes[0].set_title("Температура поверхности моря и ледовые условия в Охотском море")
axes[0].grid(True, alpha=0.3)

axes[1].plot(df["year"], df["Ice_max_Okhotsk_1e4km2_JMA"])
axes[1].set_ylabel("Макс. лёд, 10⁴ км²")
axes[1].set_xlabel("Год")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
savefig(fig, "рис3_SST_и_лед_1963_2025.png")

# Рис. 4 — PDO и AO
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(df["year"], df["PDO_DJFM"])
axes[0].axhline(0, linestyle="--", linewidth=1)
axes[0].set_ylabel("PDO (DJFM)")
axes[0].set_title("Крупномасштабные климатические индексы: PDO и AO (DJFM)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(df["year"], df["AO_DJFM"])
axes[1].axhline(0, linestyle="--", linewidth=1)
axes[1].set_ylabel("AO (DJFM)")
axes[1].set_xlabel("Год")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
savefig(fig, "рис4_PDO_AO_1963_2025.png")

# Рис. 5 — сравнение режимов (boxplot)
sub = df[(df.year >= 1971) & (df.year <= 2020)].dropna(
    subset=["SST_anom_degC", "SSB_million_t_updated", "catch_total_thousand_t"]
).copy()
warm = sub[sub["SST_anom_degC"] > 0]
cold = sub[sub["SST_anom_degC"] <= 0]

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
axes[0].boxplot([warm["SSB_million_t_updated"], cold["SSB_million_t_updated"]],
                labels=["Тёплый", "Холодный"])
axes[0].set_ylabel("SSB, млн т")
axes[0].set_title("Сравнение режимов (1971–2020): биомасса и вылов")
axes[0].grid(True, axis="y", alpha=0.3)

axes[1].boxplot([warm["catch_total_thousand_t"], cold["catch_total_thousand_t"]],
                labels=["Тёплый", "Холодный"])
axes[1].set_ylabel("Вылов, тыс. т")
axes[1].set_xlabel("Режим по SST")
axes[1].grid(True, axis="y", alpha=0.3)

fig.tight_layout()
savefig(fig, "рис5_режимы_boxplot_SSB_вылов.png")

# Рис. 6 — остатки модели Ricker
# Подготовка таблицы с ковариатами года нереста (t-2)
r = df[["year", "Recruitment_billion_individuals_ilyin2016"]].dropna().copy()
r["spawn_year"] = r["year"] - 2

spawn = df[["year", "SSB_million_t_updated", "Ice_max_Okhotsk_1e4km2_JMA", "SST_anom_degC",
            "AO_DJFM", "PDO_DJFM"]].rename(columns={"year": "spawn_year"})
r = r.merge(spawn, on="spawn_year", how="left").dropna(
    subset=["SSB_million_t_updated", "Ice_max_Okhotsk_1e4km2_JMA", "SST_anom_degC", "AO_DJFM", "PDO_DJFM"]
)
r = r[(r.year >= 1973) & (r.year <= 2014)].copy()

r["log_R_over_SSB"] = np.log(r["Recruitment_billion_individuals_ilyin2016"] / r["SSB_million_t_updated"])
m_base = sm.OLS(r["log_R_over_SSB"], sm.add_constant(r[["SSB_million_t_updated"]])).fit()
r["resid"] = m_base.resid

m_ice = sm.OLS(r["resid"], sm.add_constant(r[["Ice_max_Okhotsk_1e4km2_JMA"]])).fit()
m_sst = sm.OLS(r["resid"], sm.add_constant(r[["SST_anom_degC"]])).fit()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(r["Ice_max_Okhotsk_1e4km2_JMA"], r["resid"])
x = np.linspace(r["Ice_max_Okhotsk_1e4km2_JMA"].min(), r["Ice_max_Okhotsk_1e4km2_JMA"].max(), 100)
axes[0].plot(x, m_ice.params["const"] + m_ice.params["Ice_max_Okhotsk_1e4km2_JMA"] * x)
axes[0].set_xlabel("Макс. лёд, 10⁴ км² (год нереста)")
axes[0].set_ylabel("Остатки (log(R/SSB))")
axes[0].set_title("Остатки vs лёд")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(r["SST_anom_degC"], r["resid"])
x = np.linspace(r["SST_anom_degC"].min(), r["SST_anom_degC"].max(), 100)
axes[1].plot(x, m_sst.params["const"] + m_sst.params["SST_anom_degC"] * x)
axes[1].set_xlabel("Аномалия SST, °C (год нереста)")
axes[1].set_title("Остатки vs SST")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Климатический сигнал в пополнении: остатки базовой модели Ricker", y=1.02)
fig.tight_layout()
savefig(fig, "рис6_ricker_остатки_лед_SST.png")
