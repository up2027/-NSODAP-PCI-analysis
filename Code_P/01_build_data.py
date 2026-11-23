# Build the harmonized, weighted analysis file.
# Input (data/): NSODAP2012.xlsx, NSODAP2019.xlsx, NSODAP2025.xlsx
# Output (output/): analysis.pkl, raw2025.pkl, 01_build_check.xlsx
import numpy as np
import pandas as pd

TXT4 = {"Strongly Agree": 1, "Somewhat Agree": 2, "Somewhat Disagree": 3, "Strongly Disagree": 4}
SUP4 = {"Strongly support": 1, "Support": 2, "Oppose": 3, "Strongly oppose": 4}


def agree_score(x, reverse_2012=False):
    # 1-4, 4 = strongly agree; missing for don't know or refused
    x = pd.to_numeric(x, errors="coerce")
    x = x.where(x.between(1, 4))
    return x if reverse_2012 else 5 - x


def col(df, prefix):
    hits = [c for c in df.columns if str(c).startswith(prefix)]
    return df[hits[0]]


r12 = pd.read_excel("data/NSODAP2012.xlsx")
r19 = pd.read_excel("data/NSODAP2019.xlsx")
r25 = pd.read_excel("data/NSODAP2025.xlsx")

# 2012 (scales coded 4 = strongly agree)
a = pd.DataFrame({"year": 2012, "wt": r12["finalwt"], "mode": "phone"})
a["registered"] = ((r12[["Q13A", "Q13B", "Q13C"]] == 1).any(axis=1)).astype(float)
a["support4"] = agree_score(r12["Q4"], reverse_2012=True)
a["doc4"] = agree_score(r12["Q16G"], True)
a["fair4"] = agree_score(r12["Q16M"], True)
a["burial4"] = agree_score(r12["Q12A"], True)
a["disfig4"] = agree_score(r12["Q16B"], True)
a["pc_support4"] = agree_score(r12["Q17"], True)
a["optout"] = r12["Q17C"].map({1: 1.0, 2: 0.0})
d9 = r12["D9"]
a["inc4"] = np.select([d9.between(1, 3), d9.between(4, 5), d9.between(6, 7), d9.between(8, 12)], [1, 2, 3, 4], np.nan)
a["hispanic"] = r12["ethnicity_cat"].map({1: 1.0, 2: 0.0})

# 2019
b = pd.DataFrame({"year": 2019, "wt": r19["WTFINAL"], "mode": r19["Mode"].map({1: "phone", 2: "web"})})
b["wt_mode"] = np.where(r19["Mode"] == 1, r19["WTPHONE"], r19["WTWEB"])
b["registered"] = (r19["Q13"] == 1).astype(float)
b["support4"] = agree_score(r19["Q4"])
b["doc4"] = agree_score(r19["Q16G"])
b["fair4"] = agree_score(r19["Q16K"])
b["burial4"] = agree_score(r19["Q12A"])
b["disfig4"] = agree_score(r19["Q16B"])
b["pc_support4"] = agree_score(r19["Q17"])
b["optout"] = r19["Q17C"].map({1: 1.0, 2: 0.0})
d9 = r19["D9"]
b["inc4"] = np.select([d9 == 1, d9.between(2, 3), d9.between(4, 6), d9.between(7, 8)], [1, 2, 3, 4], np.nan)
b["hispanic"] = r19["Hispanic"].astype(float)


def txt(s, m):
    return 5 - s.map(m)


c = pd.DataFrame({"year": 2025, "wt": col(r25, "WTFINAL")})
c["mode"] = np.where(col(r25, "QS1b. Have I reached you").notna(), "phone", "web")
c["wt_mode"] = c["wt"]
c["registered"] = (col(r25, "Q13. Are you signed up") == "Yes").astype(float)
c["support4"] = txt(col(r25, "Q4. In general"), SUP4)
c["doc4"] = txt(col(r25, "Q12L. If you indicate"), TXT4)
c["fair4"] = txt(col(r25, "Q12N. The U.S. transplant"), TXT4)
c["burial4"] = txt(col(r25, "Q12A. It is important for a person"), TXT4)
c["disfig4"] = txt(col(r25, "Q12J. You are worried"), TXT4)
c["pc_support4"] = txt(col(r25, "Q17. Would you"), SUP4)
c["optout"] = col(r25, "Q17c. If the United States").map({"Yes": 1.0, "No": 0.0})
inc = col(r25, "QD9. What is your total ANNUAL").astype(str)
c["inc4"] = np.select(
    [inc.str.startswith("Less than $20"), inc.str.contains(r"\$20,000|\$30,000 or more"),
     inc.str.contains(r"\$40,000|\$50,000|\$60,000"), inc.str.contains(r"\$75,000|\$100,000|\$150,000")],
    [1, 2, 3, 4], np.nan)
c["hispanic"] = (col(r25, "Hispanic") == "Hispanic").astype(float)

raw = pd.concat([a, b, c], ignore_index=True)

# Race, age group, sex and education
r25c = lambda p: col(r25, p)
race = pd.concat([
    r12["race_cat"].map({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5}),
    r19["RaceCat"].map({1: 1, 2: 2, 3: 3, 4: 4, 5: 4}),
    r25c("Race Category").map({"Race/Ethnicity: White": 1, "Race/Ethnicity: Black": 2, "Race/Ethnicity: American Indian": 3,
                               "Race/Ethnicity: Asian": 4, "Race/Ethnicity: Hispanic": 5, "Race/Ethnicity: Multiple/Other": 5})],
    ignore_index=True)
age = pd.concat([r12["age_cat"], r19["AgeCat"].where(r19["AgeCat"].between(1, 4)),
                 r25c("Age Category").map({"Age: 18 - 34": 1, "Age: 35 - 49": 2, "Age: 50 - 64": 3, "Age: 65+": 4})], ignore_index=True)
sex = pd.concat([r12["D1"], r19["D1"].where(r19["D1"].isin([1, 2])),
                 r25c("Sex, 3 levels").map({"Male": 1, "Female": 2})], ignore_index=True)
edu = pd.concat([r12["D3"].map({1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4}), r19["EduCat"],
                 r25c("Educational Category").map({"Education: Up to High School": 1, "Education: Some College/Trade School": 2,
                                                   "Education: College Bachelors": 3, "Education: Postgraduate": 4})],
                ignore_index=True)
checks = []
raw["race"] = race.map({1: "White", 2: "Black", 3: "NativeAm", 4: "Asian", 5: "Other"}).values
# Race groups exclude Hispanic respondents in every wave
raw["race_nh"] = np.where(raw["hispanic"] == 1, "Hispanic", raw["race"])
raw["age"] = age.values
raw["female"] = sex.map({1: 0.0, 2: 1.0}).values
raw["edu"] = edu.values

# Concern scores (1-4, higher = more concern)
raw["mistrust"] = raw[["doc4"]].assign(f=5 - raw["fair4"]).mean(axis=1, skipna=False)
raw["bodyint"] = raw[["burial4", "disfig4"]].mean(axis=1, skipna=False)
raw["supporter"] = (raw["support4"] >= 3).astype(float).where(raw["support4"].notna())
raw["wt_norm"] = raw["wt"] / raw.groupby("year")["wt"].transform("mean")

checks.append(f"N = {len(raw)}; by year: {raw.year.value_counts().sort_index().to_dict()}")
print("\n".join(checks))

# Save
raw.to_pickle("output/analysis.pkl")
r25.to_pickle("output/raw2025.pkl")
with pd.ExcelWriter("output/01_build_check.xlsx") as xw:
    pd.DataFrame({"check": checks}).to_excel(xw, sheet_name="checks", index=False)
    pd.crosstab(raw.race, raw.year).to_excel(xw, sheet_name="race_by_year")
    pd.crosstab(raw.race_nh, raw.year).to_excel(xw, sheet_name="race_nh_by_year")
