"""
Synthetic B2B Sales CRM data generator.

Produces a realistic relational dataset:
  territories, sales_reps, products, accounts, contacts, leads,
  opportunities, pipeline_history, sales_activities, quotes,
  won_deals, lost_deals

Design goals:
  - Embed realistic correlations between company size, industry, engagement,
    pricing/discounting, sales activity, deal age and conversion, so that
    downstream lead scoring / forecasting / survival models have real signal
    to find (not just noise).
  - Fully vectorized (numpy/pandas) so ~100K accounts / 300K+ opportunities /
    1M+ activities generate in seconds, not hours.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

RNG = np.random.default_rng(42)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)

AS_OF_DATE = pd.Timestamp("2026-09-18")  # "today" snapshot for the dataset
HISTORY_START = pd.Timestamp("2024-01-01")  # ~2.75 years of CRM history
LEAD_WINDOW_END = pd.Timestamp("2026-08-15")  # leave a buffer so recent leads can still be open

N_TERRITORIES = 24
N_REPS = 180
N_PRODUCTS = 16
N_ACCOUNTS = 120_000
N_LEADS = 1_000_000  # ~30% convert -> 300K+ opportunities

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def random_dates(start, end, n, rng=RNG):
    start_u = start.value // 10**9
    end_u = end.value // 10**9
    return pd.to_datetime(rng.integers(start_u, end_u, size=n), unit="s")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ----------------------------------------------------------------------------
# Territories
# ----------------------------------------------------------------------------
REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
region_weights = [0.45, 0.30, 0.18, 0.07]

territories = pd.DataFrame({
    "territory_id": np.arange(1, N_TERRITORIES + 1),
    "territory_name": [f"Territory-{i:02d}" for i in range(1, N_TERRITORIES + 1)],
})
territories["region"] = RNG.choice(REGIONS, size=N_TERRITORIES, p=region_weights)
country_map = {
    "North America": ["USA", "Canada"],
    "EMEA": ["UK", "Germany", "France", "UAE", "South Africa"],
    "APAC": ["Australia", "Singapore", "India", "Japan"],
    "LATAM": ["Brazil", "Mexico", "Chile"],
}
territories["country"] = territories["region"].apply(lambda r: RNG.choice(country_map[r]))

territories.to_csv(f"{OUT_DIR}/territories.csv", index=False)

# ----------------------------------------------------------------------------
# Sales reps
# ----------------------------------------------------------------------------
SENIORITY = ["Junior", "Mid", "Senior", "Principal"]
seniority_p = [0.30, 0.35, 0.25, 0.10]
seniority_skill = {"Junior": -0.35, "Mid": 0.0, "Senior": 0.30, "Principal": 0.55}

rep_ids = np.arange(1, N_REPS + 1)
rep_territory = RNG.choice(territories["territory_id"], size=N_REPS)
rep_seniority = RNG.choice(SENIORITY, size=N_REPS, p=seniority_p)
rep_hire_date = random_dates(pd.Timestamp("2018-01-01"), pd.Timestamp("2026-06-01"), N_REPS)
first_names = ["James","Maria","Ahmed","Wei","Priya","John","Fatima","Carlos","Olga","Sam",
               "Lena","Omar","Grace","Ivan","Noah","Aisha","Leo","Mei","Tariq","Sofia"]
last_names = ["Smith","Khan","Garcia","Chen","Patel","Müller","Silva","Kowalski","Nguyen",
              "Brown","Rossi","Kim","Diallo","Novak","Andrade"]
rep_names = [f"{RNG.choice(first_names)} {RNG.choice(last_names)}" for _ in range(N_REPS)]

sales_reps = pd.DataFrame({
    "rep_id": rep_ids,
    "rep_name": rep_names,
    "territory_id": rep_territory,
    "seniority_level": rep_seniority,
    "hire_date": rep_hire_date.date,
    "quota_target_quarterly": (RNG.normal(180_000, 40_000, N_REPS)
                                 + np.vectorize(seniority_skill.get)(rep_seniority) * 60_000).round(-3).clip(60_000, None),
})
sales_reps.to_csv(f"{OUT_DIR}/sales_reps.csv", index=False)

rep_skill_lookup = np.vectorize(seniority_skill.get)(rep_seniority)  # aligned to rep_ids-1

# ----------------------------------------------------------------------------
# Products
# ----------------------------------------------------------------------------
PRODUCT_CATEGORIES = ["Platform License", "Add-on Module", "Professional Services",
                       "Support & Success", "Data & Analytics"]
product_ids = np.arange(1, N_PRODUCTS + 1)
product_cat = RNG.choice(PRODUCT_CATEGORIES, size=N_PRODUCTS,
                          p=[0.35, 0.25, 0.15, 0.15, 0.10])
base_price = {
    "Platform License": (15_000, 120_000),
    "Add-on Module": (3_000, 25_000),
    "Professional Services": (5_000, 60_000),
    "Support & Success": (2_000, 20_000),
    "Data & Analytics": (8_000, 70_000),
}
list_price = np.array([RNG.uniform(*base_price[c]) for c in product_cat]).round(-2)
products = pd.DataFrame({
    "product_id": product_ids,
    "product_name": [f"{c.split()[0]}-{i}" for i, c in zip(product_ids, product_cat)],
    "category": product_cat,
    "list_price": list_price,
})
products.to_csv(f"{OUT_DIR}/products.csv", index=False)

# ----------------------------------------------------------------------------
# Accounts (companies)
# ----------------------------------------------------------------------------
INDUSTRIES = ["Technology", "Manufacturing", "Healthcare", "Financial Services",
              "Retail & eCommerce", "Education", "Government", "Professional Services",
              "Energy & Utilities", "Media & Telecom"]
industry_p = [0.18, 0.13, 0.11, 0.12, 0.11, 0.07, 0.05, 0.10, 0.06, 0.07]

SIZE_BANDS = ["SMB", "Mid-Market", "Enterprise"]
size_p = [0.55, 0.32, 0.13]
size_employee_range = {"SMB": (5, 250), "Mid-Market": (251, 2500), "Enterprise": (2501, 60000)}
# base win-propensity / deal-size multiplier by size band (larger co = bigger deals, slightly
# longer cycles, but also more stable budget => modestly higher eventual win rate)
size_deal_mult = {"SMB": 1.0, "Mid-Market": 2.8, "Enterprise": 7.5}
size_win_bonus = {"SMB": -0.05, "Mid-Market": 0.05, "Enterprise": 0.15}

industry_win_bonus = {
    "Technology": 0.10, "Manufacturing": 0.00, "Healthcare": 0.05,
    "Financial Services": 0.08, "Retail & eCommerce": -0.05, "Education": -0.10,
    "Government": -0.15, "Professional Services": 0.02, "Energy & Utilities": 0.00,
    "Media & Telecom": -0.02,
}

account_ids = np.arange(1, N_ACCOUNTS + 1)
acc_industry = RNG.choice(INDUSTRIES, size=N_ACCOUNTS, p=industry_p)
acc_size = RNG.choice(SIZE_BANDS, size=N_ACCOUNTS, p=size_p)
acc_employees = np.array([RNG.integers(*size_employee_range[s]) for s in acc_size])
acc_revenue = (acc_employees * RNG.uniform(80_000, 260_000, N_ACCOUNTS)).round(-3)
acc_territory = RNG.choice(territories["territory_id"], size=N_ACCOUNTS)
acc_created = random_dates(HISTORY_START - pd.Timedelta(days=730), HISTORY_START, N_ACCOUNTS)
# engagement score 0-100: proxy for how responsive/active this account tends to be
acc_engagement = np.clip(RNG.normal(55, 20, N_ACCOUNTS), 1, 100).round(1)

company_suffixes = ["Inc.", "LLC", "Group", "Holdings", "Partners", "Co.", "Ltd.", "Solutions"]
company_stems = ["Nova", "Summit", "Bluewave", "Cedar", "Vertex", "Orbit", "Granite", "Lumen",
                 "Harbor", "Falcon", "Atlas", "Meridian", "Cobalt", "Anchor", "Pinnacle",
                 "Vantage", "Northgate", "Crestline", "Ironwood", "Silverline"]
acc_names = [f"{RNG.choice(company_stems)} {RNG.choice(company_suffixes)} {i}" for i in range(N_ACCOUNTS)]

accounts = pd.DataFrame({
    "account_id": account_ids,
    "account_name": acc_names,
    "industry": acc_industry,
    "company_size_band": acc_size,
    "employees": acc_employees,
    "annual_revenue_usd": acc_revenue,
    "territory_id": acc_territory,
    "created_date": acc_created.date,
    "engagement_score": acc_engagement,
})
accounts.to_csv(f"{OUT_DIR}/accounts.csv", index=False)

# lookups aligned by position (account_id - 1)
acc_size_mult = np.vectorize(size_deal_mult.get)(acc_size)
acc_size_bonus = np.vectorize(size_win_bonus.get)(acc_size)
acc_ind_bonus = np.vectorize(industry_win_bonus.get)(acc_industry)

print(f"accounts: {len(accounts):,}")

# ----------------------------------------------------------------------------
# Contacts
# ----------------------------------------------------------------------------
TITLES = ["Buyer/Manager", "Director", "VP", "C-Level", "Individual Contributor"]
title_p = [0.30, 0.28, 0.20, 0.10, 0.12]
title_seniority_score = {"Individual Contributor": 0, "Buyer/Manager": 1, "Director": 2, "VP": 3, "C-Level": 4}

n_contacts_per_account = RNG.integers(1, 4, N_ACCOUNTS)  # 1-3 contacts
contact_account_id = np.repeat(account_ids, n_contacts_per_account)
N_CONTACTS = len(contact_account_id)
contact_ids = np.arange(1, N_CONTACTS + 1)
contact_title = RNG.choice(TITLES, size=N_CONTACTS, p=title_p)
contact_created_offset = RNG.integers(0, 60, N_CONTACTS)
acc_created_lookup = pd.to_datetime(accounts.set_index("account_id")["created_date"]).reindex(contact_account_id).values
contact_created = pd.to_datetime(acc_created_lookup) + pd.to_timedelta(contact_created_offset, unit="D")

contact_first = RNG.choice(first_names, size=N_CONTACTS)
contact_last = RNG.choice(last_names, size=N_CONTACTS)

contacts = pd.DataFrame({
    "contact_id": contact_ids,
    "account_id": contact_account_id,
    "contact_name": [f"{f} {l}" for f, l in zip(contact_first, contact_last)],
    "title": contact_title,
    "seniority_score": np.vectorize(title_seniority_score.get)(contact_title),
    "email": [f"contact{i}@example-mail.com" for i in contact_ids],
    "created_date": contact_created.date,
})
contacts.to_csv(f"{OUT_DIR}/contacts.csv", index=False)
print(f"contacts: {len(contacts):,}")

# ----------------------------------------------------------------------------
# Leads
# ----------------------------------------------------------------------------
SOURCES = ["Inbound - Web", "Outbound - Cold Call", "Outbound - Email", "Referral",
           "Partner Channel", "Trade Show / Event"]
source_p = [0.28, 0.16, 0.14, 0.16, 0.12, 0.14]
source_conv_bonus = {
    "Inbound - Web": 0.08, "Outbound - Cold Call": -0.12, "Outbound - Email": -0.05,
    "Referral": 0.18, "Partner Channel": 0.10, "Trade Show / Event": 0.02,
}

lead_account_id = RNG.choice(account_ids, size=N_LEADS)
# bias contact selection to a contact from the same account
contacts_by_account = contacts.groupby("account_id")["contact_id"].apply(list)

lead_created = random_dates(HISTORY_START, LEAD_WINDOW_END, N_LEADS)
lead_source = RNG.choice(SOURCES, size=N_LEADS, p=source_p)

lead_acc_engagement = accounts.set_index("account_id")["engagement_score"].reindex(lead_account_id).values
lead_acc_size_bonus = pd.Series(acc_size_bonus, index=account_ids).reindex(lead_account_id).values
lead_acc_ind_bonus = pd.Series(acc_ind_bonus, index=account_ids).reindex(lead_account_id).values
lead_source_bonus = np.vectorize(source_conv_bonus.get)(lead_source)

conv_logit = (-0.9
              + (lead_acc_engagement - 55) / 100
              + lead_acc_size_bonus
              + lead_acc_ind_bonus * 0.5
              + lead_source_bonus
              + RNG.normal(0, 0.55, N_LEADS))
conv_prob = sigmoid(conv_logit)
converted = RNG.random(N_LEADS) < conv_prob

lead_score = np.clip((conv_prob * 100) + RNG.normal(0, 8, N_LEADS), 1, 99).round(1)
lead_status = np.where(converted, "Converted", RNG.choice(["Disqualified", "Nurturing"], size=N_LEADS, p=[0.6, 0.4]))

leads = pd.DataFrame({
    "lead_id": np.arange(1, N_LEADS + 1),
    "account_id": lead_account_id,
    "source": lead_source,
    "created_date": lead_created.date,
    "lead_score": lead_score,
    "status": lead_status,
    "converted_to_opportunity": converted,
})
leads.to_csv(f"{OUT_DIR}/leads.csv", index=False)
print(f"leads: {len(leads):,} | converted: {converted.sum():,}")

# assign a contact per lead (same account)
def pick_contact(acc_id):
    lst = contacts_by_account.get(acc_id)
    return RNG.choice(lst) if lst else np.nan

leads_contact_map = {a: contacts_by_account.get(a, [np.nan]) for a in np.unique(lead_account_id)}
lead_contact_id = np.array([RNG.choice(leads_contact_map[a]) for a in lead_account_id])
leads["contact_id"] = lead_contact_id
leads.to_csv(f"{OUT_DIR}/leads.csv", index=False)

del leads_contact_map

# ----------------------------------------------------------------------------
# Opportunities
# ----------------------------------------------------------------------------
opp_leads = leads[leads["converted_to_opportunity"]].reset_index(drop=True)
N_OPP = len(opp_leads)

opp_ids = np.arange(1, N_OPP + 1)
lead_to_opp_lag = RNG.integers(1, 35, N_OPP)
opp_created = pd.to_datetime(opp_leads["created_date"]) + pd.to_timedelta(lead_to_opp_lag, unit="D")
opp_created = pd.Series(pd.to_datetime(opp_created.dt.date))

opp_account_id = opp_leads["account_id"].values
opp_contact_id = opp_leads["contact_id"].values
opp_lead_id = opp_leads["lead_id"].values
opp_source = opp_leads["source"].values

acc_territory_lookup = accounts.set_index("account_id")["territory_id"]
opp_territory = acc_territory_lookup.reindex(opp_account_id).values
reps_by_territory = sales_reps.groupby("territory_id")["rep_id"].apply(list)
all_rep_ids = sales_reps["rep_id"].values
opp_rep_id = np.array([
    RNG.choice(reps_by_territory[t]) if t in reps_by_territory.index else RNG.choice(all_rep_ids)
    for t in opp_territory
])

opp_product_id = RNG.choice(product_ids, size=N_OPP)
opp_list_price = products.set_index("product_id")["list_price"].reindex(opp_product_id).values

acc_idx = accounts.set_index("account_id")
opp_size_band = acc_idx["company_size_band"].reindex(opp_account_id).values
opp_industry = acc_idx["industry"].reindex(opp_account_id).values
opp_engagement = acc_idx["engagement_score"].reindex(opp_account_id).values
opp_size_mult = pd.Series(acc_size_mult, index=account_ids).reindex(opp_account_id).values
opp_size_bonus = pd.Series(acc_size_bonus, index=account_ids).reindex(opp_account_id).values
opp_ind_bonus = pd.Series(acc_ind_bonus, index=account_ids).reindex(opp_account_id).values
opp_source_bonus = np.vectorize(source_conv_bonus.get)(opp_source)
opp_rep_skill = pd.Series(rep_skill_lookup, index=rep_ids).reindex(opp_rep_id).values

deal_noise = RNG.lognormal(mean=0, sigma=0.35, size=N_OPP)
deal_qty = RNG.integers(1, 4, N_OPP)
deal_value = (opp_list_price * deal_qty * opp_size_mult * deal_noise).round(-2)
deal_value = np.clip(deal_value, 500, None)

size_stakeholder_lambda = {"SMB": 1.6, "Mid-Market": 3.0, "Enterprise": 5.0}
lam_stake = np.vectorize(size_stakeholder_lambda.get)(opp_size_band)
num_stakeholders = RNG.poisson(lam_stake) + 1

log_deal = np.log(deal_value)
deal_value_z = (log_deal - log_deal.mean()) / log_deal.std()
discount_requested_pct = np.clip(
    8 + (opp_size_band == "Enterprise") * 6 + (opp_size_band == "Mid-Market") * 3
    + deal_value_z * 3 + RNG.normal(0, 5, N_OPP), 0, 45).round(1)

engagement_z = (opp_engagement - 55) / 20
activity_intensity = sigmoid(0.9 * engagement_z + 0.6 * opp_rep_skill + RNG.normal(0, 0.5, N_OPP))

base_cycle = {"SMB": 35, "Mid-Market": 65, "Enterprise": 110}
cycle_mu = np.vectorize(base_cycle.get)(opp_size_band) + num_stakeholders * 4
cycle_length = np.clip(RNG.lognormal(mean=np.log(cycle_mu), sigma=0.35), 7, 400).round().astype(int)

stakeholder_effect = -0.06 * np.abs(num_stakeholders - 3)
discount_effect = np.clip(discount_requested_pct / 45, 0, 1) * 0.5 - 0.05
size_effect = -0.045 * np.clip(deal_value_z, 0, None)

win_logit = (-0.55
             + engagement_z * 0.35
             + opp_size_bonus
             + opp_ind_bonus * 0.6
             + opp_source_bonus
             + opp_rep_skill * 0.55
             + (activity_intensity - 0.5) * 1.3
             + stakeholder_effect
             + discount_effect
             + size_effect
             + RNG.normal(0, 0.6, N_OPP))
true_win_prob = sigmoid(win_logit)

scheduled_close_date = opp_created + pd.to_timedelta(cycle_length, unit="D")
would_win = RNG.random(N_OPP) < true_win_prob
is_closed = (scheduled_close_date <= AS_OF_DATE).values

close_status = np.where(~is_closed, "Open", np.where(would_win, "Closed Won", "Closed Lost"))
actual_close_date = scheduled_close_date.where(pd.Series(is_closed), pd.NaT)

STAGE_SEQ = ["Qualification", "Needs Analysis", "Proposal", "Negotiation"]
stage_cum_fracs = np.array([0.15, 0.40, 0.70, 1.0])
days_elapsed = np.where(is_closed, cycle_length, (AS_OF_DATE - opp_created).dt.days.values)
frac_elapsed = np.clip(days_elapsed / np.maximum(cycle_length, 1), 0, 1)

stage_idx = np.searchsorted(stage_cum_fracs, frac_elapsed, side="left")
stage_idx = np.clip(stage_idx, 0, len(STAGE_SEQ) - 1)
current_stage_open = np.array(STAGE_SEQ)[stage_idx]

lost_stage_shrink = RNG.integers(0, 2, N_OPP)
lost_final_stage_idx = np.clip(stage_idx - lost_stage_shrink, 0, len(STAGE_SEQ) - 1)
lost_final_stage = np.array(STAGE_SEQ)[lost_final_stage_idx]

stage = np.select(
    [close_status == "Open", close_status == "Closed Won", close_status == "Closed Lost"],
    [current_stage_open, np.full(N_OPP, "Closed Won"), np.full(N_OPP, "Closed Lost")],
    default="Unknown",
)
stage_reached_before_close = np.where(close_status == "Closed Lost", lost_final_stage, current_stage_open)
final_stage_idx_for_history = np.where(close_status == "Closed Lost", lost_final_stage_idx, stage_idx)

LOST_REASONS = ["Budget Constraints", "Chose Competitor", "No Decision / Timing",
                "Feature Gap", "Pricing Too High", "Champion Left Company"]
lost_reason = np.where(close_status == "Closed Lost", RNG.choice(LOST_REASONS, size=N_OPP), None)

opportunities = pd.DataFrame({
    "opportunity_id": opp_ids,
    "account_id": opp_account_id,
    "contact_id": opp_contact_id,
    "lead_id": opp_lead_id,
    "rep_id": opp_rep_id,
    "product_id": opp_product_id,
    "source": opp_source,
    "created_date": opp_created.dt.date,
    "deal_value": deal_value,
    "discount_requested_pct": discount_requested_pct,
    "num_stakeholders": num_stakeholders,
    "expected_close_date": scheduled_close_date.dt.date,
    "actual_close_date": actual_close_date.dt.date,
    "stage": stage,
    "stage_reached_before_close": stage_reached_before_close,
    "close_status": close_status,
    "lost_reason": lost_reason,
    "sales_cycle_days_planned": cycle_length,
})
opportunities.to_csv(f"{OUT_DIR}/opportunities.csv", index=False)
print(f"opportunities: {len(opportunities):,} | won={sum(close_status=='Closed Won'):,} "
      f"lost={sum(close_status=='Closed Lost'):,} open={sum(close_status=='Open'):,}")

# Save internal arrays needed by later steps (not part of public schema)
np.save(f"{OUT_DIR}/_activity_intensity.npy", activity_intensity)

# ----------------------------------------------------------------------------
# Pipeline history (stage-by-stage transitions per opportunity)
# ----------------------------------------------------------------------------
n_stages_per_opp = (final_stage_idx_for_history + 1).astype(int)
total_hist_rows = int(n_stages_per_opp.sum())

group_starts = np.cumsum(n_stages_per_opp) - n_stages_per_opp
hist_opp_idx = np.repeat(np.arange(N_OPP), n_stages_per_opp)  # 0-based index into opportunities
stage_position = np.arange(total_hist_rows) - np.repeat(group_starts, n_stages_per_opp)

hist_opportunity_id = opp_ids[hist_opp_idx]
hist_stage_name = np.array(STAGE_SEQ)[stage_position]
hist_cycle = cycle_length[hist_opp_idx]
hist_created = opp_created.values[hist_opp_idx]
hist_is_closed = is_closed[hist_opp_idx]
hist_close_status = close_status[hist_opp_idx]
hist_n_stages_total = n_stages_per_opp[hist_opp_idx]
hist_actual_close = actual_close_date.values[hist_opp_idx]

entered_frac_lookup = np.concatenate([[0.0], stage_cum_fracs[:-1]])
entered_frac = entered_frac_lookup[stage_position]
exited_frac = stage_cum_fracs[stage_position]

entered_date = pd.Series(hist_created) + pd.to_timedelta((entered_frac * hist_cycle).round(), unit="D")
exited_date_calc = pd.Series(hist_created) + pd.to_timedelta((exited_frac * hist_cycle).round(), unit="D")

is_last_row = (stage_position == hist_n_stages_total - 1)
still_open_row = is_last_row & (~hist_is_closed)

exited_date = exited_date_calc.where(~still_open_row, pd.NaT)
# For a closed opportunity's final funnel stage, snap the exit to the actual close date
exited_date = exited_date.where(~(is_last_row & hist_is_closed), pd.Series(hist_actual_close))

duration_days = (exited_date - entered_date).dt.days
elapsed_days_if_open = (AS_OF_DATE - entered_date).dt.days.where(still_open_row)

pipeline_history = pd.DataFrame({
    "history_id": np.arange(1, total_hist_rows + 1),
    "opportunity_id": hist_opportunity_id,
    "stage": hist_stage_name,
    "entered_date": entered_date.dt.date,
    "exited_date": exited_date.dt.date,
    "duration_days": duration_days,
    "still_in_stage": still_open_row,
    "days_in_stage_so_far": elapsed_days_if_open,
})
pipeline_history.to_csv(f"{OUT_DIR}/pipeline_history.csv", index=False)
print(f"pipeline_history: {len(pipeline_history):,}")

# ----------------------------------------------------------------------------
# Sales activities
# ----------------------------------------------------------------------------
lam_activities = np.clip(1.4 + activity_intensity * 6.5 + cycle_length / 55, 0.3, None)
n_activities_per_opp = RNG.poisson(lam_activities)
n_activities_per_opp = np.maximum(n_activities_per_opp, 1)  # every opportunity gets at least 1 touch
total_act_rows = int(n_activities_per_opp.sum())

act_group_starts = np.cumsum(n_activities_per_opp) - n_activities_per_opp
act_opp_idx = np.repeat(np.arange(N_OPP), n_activities_per_opp)
act_position = np.arange(total_act_rows) - np.repeat(act_group_starts, n_activities_per_opp)

act_opportunity_id = opp_ids[act_opp_idx]
act_rep_id = opp_rep_id[act_opp_idx]
act_created = opp_created.values[act_opp_idx]
act_is_closed = is_closed[act_opp_idx]
act_close_date = actual_close_date.values[act_opp_idx]
act_intensity = activity_intensity[act_opp_idx]
act_cycle = cycle_length[act_opp_idx]

# activity window end = close date if closed, else AS_OF_DATE
window_end = np.where(act_is_closed, act_close_date, np.datetime64(AS_OF_DATE))
window_days = (pd.Series(window_end) - pd.Series(act_created)).dt.days.clip(lower=1).values

# U-shaped distribution across the window (more touches early + near close), Beta(0.5,0.5)
beta_frac = RNG.beta(0.6, 0.6, size=total_act_rows)
activity_offset_days = (beta_frac * window_days).round().astype(int)
activity_date = pd.Series(act_created) + pd.to_timedelta(activity_offset_days, unit="D")

ACTIVITY_TYPES = ["Call", "Email", "Meeting", "Demo", "Proposal Discussion"]
activity_type_p = [0.34, 0.32, 0.16, 0.10, 0.08]
activity_type = RNG.choice(ACTIVITY_TYPES, size=total_act_rows, p=activity_type_p)

outcome_positive_prob = np.clip(0.25 + act_intensity * 0.5, 0.05, 0.9)
outcome_roll = RNG.random(total_act_rows)
OUTCOMES_POS = "Positive - Advancing"
OUTCOMES_NEU = "Neutral"
OUTCOMES_NEG = "Negative / Objection"
OUTCOMES_NOR = "No Response"
outcome = np.select(
    [outcome_roll < outcome_positive_prob,
     outcome_roll < outcome_positive_prob + 0.35 * (1 - outcome_positive_prob),
     outcome_roll < outcome_positive_prob + 0.7 * (1 - outcome_positive_prob)],
    [OUTCOMES_POS, OUTCOMES_NEU, OUTCOMES_NEG],
    default=OUTCOMES_NOR,
)
duration_minutes = np.where(
    np.isin(activity_type, ["Call", "Meeting", "Demo"]),
    np.clip(RNG.normal(28, 14, total_act_rows), 5, 120).round(),
    0,
)

sales_activities = pd.DataFrame({
    "activity_id": np.arange(1, total_act_rows + 1),
    "opportunity_id": act_opportunity_id,
    "rep_id": act_rep_id,
    "activity_type": activity_type,
    "activity_date": activity_date.dt.date,
    "duration_minutes": duration_minutes.astype(int),
    "outcome": outcome,
})
sales_activities.to_csv(f"{OUT_DIR}/sales_activities.csv", index=False)
print(f"sales_activities: {len(sales_activities):,}")

# ----------------------------------------------------------------------------
# Quotes (issued once an opportunity reaches Proposal stage or later)
# ----------------------------------------------------------------------------
reached_proposal = final_stage_idx_for_history >= 2  # Proposal(2) or Negotiation(3)
quote_opp_mask_idx = np.where(reached_proposal)[0]
# some negotiated deals get a 2nd revised quote
n_quotes = np.where(
    (final_stage_idx_for_history[quote_opp_mask_idx] == 3) & (RNG.random(len(quote_opp_mask_idx)) < 0.4),
    2, 1
)
total_quote_rows = int(n_quotes.sum())
q_group_starts = np.cumsum(n_quotes) - n_quotes
q_opp_local_idx = np.repeat(quote_opp_mask_idx, n_quotes)
q_revision = np.arange(total_quote_rows) - np.repeat(q_group_starts, n_quotes)

q_created = opp_created.values[q_opp_local_idx]
q_cycle = cycle_length[q_opp_local_idx]
proposal_enter_day = (stage_cum_fracs[1] * q_cycle).round()  # start of Proposal stage
q_date = pd.Series(q_created) + pd.to_timedelta(proposal_enter_day + q_revision * 7, unit="D")

q_deal_value = deal_value[q_opp_local_idx]
q_discount = discount_requested_pct[q_opp_local_idx]
# revised quotes trend toward a slightly larger discount
q_effective_discount = np.clip(q_discount + q_revision * 4 + RNG.normal(0, 2, total_quote_rows), 0, 55)
q_quoted_price = (q_deal_value * (1 - q_effective_discount / 100)).round(2)

q_close_status = close_status[q_opp_local_idx]
q_is_last_quote = q_revision == (np.repeat(n_quotes, n_quotes) - 1)
q_status = np.where(
    q_is_last_quote & (q_close_status == "Closed Won"), "Accepted",
    np.where(q_is_last_quote & (q_close_status == "Closed Lost"), "Rejected",
             np.where(q_is_last_quote & (q_close_status == "Open"), "Pending", "Superseded"))
)

quotes = pd.DataFrame({
    "quote_id": np.arange(1, total_quote_rows + 1),
    "opportunity_id": opp_ids[q_opp_local_idx],
    "product_id": opp_product_id[q_opp_local_idx],
    "quote_date": q_date.dt.date,
    "quoted_price": q_quoted_price,
    "discount_pct": q_effective_discount.round(1),
    "revision_number": q_revision + 1,
    "status": q_status,
})
quotes.to_csv(f"{OUT_DIR}/quotes.csv", index=False)
print(f"quotes: {len(quotes):,}")

# ----------------------------------------------------------------------------
# Won / Lost deal marts (denormalized convenience tables mirroring closed opps)
# ----------------------------------------------------------------------------
won_mask = close_status == "Closed Won"
lost_mask = close_status == "Closed Lost"

won_deals = pd.DataFrame({
    "deal_id": np.arange(1, won_mask.sum() + 1),
    "opportunity_id": opp_ids[won_mask],
    "account_id": opp_account_id[won_mask],
    "rep_id": opp_rep_id[won_mask],
    "product_id": opp_product_id[won_mask],
    "close_date": actual_close_date.values[won_mask],
    "final_value": (deal_value[won_mask] * (1 - discount_requested_pct[won_mask] / 100)).round(2),
    "list_value": deal_value[won_mask],
    "discount_pct": discount_requested_pct[won_mask],
    "sales_cycle_days": cycle_length[won_mask],
})
won_deals["close_date"] = pd.to_datetime(won_deals["close_date"]).dt.date
won_deals.to_csv(f"{OUT_DIR}/won_deals.csv", index=False)

lost_deals = pd.DataFrame({
    "deal_id": np.arange(1, lost_mask.sum() + 1),
    "opportunity_id": opp_ids[lost_mask],
    "account_id": opp_account_id[lost_mask],
    "rep_id": opp_rep_id[lost_mask],
    "product_id": opp_product_id[lost_mask],
    "close_date": actual_close_date.values[lost_mask],
    "lost_value": deal_value[lost_mask],
    "lost_reason": lost_reason[lost_mask],
    "stage_at_loss": stage_reached_before_close[lost_mask],
    "sales_cycle_days": cycle_length[lost_mask],
})
lost_deals["close_date"] = pd.to_datetime(lost_deals["close_date"]).dt.date
lost_deals.to_csv(f"{OUT_DIR}/lost_deals.csv", index=False)

print(f"won_deals: {len(won_deals):,} | lost_deals: {len(lost_deals):,}")
print("\nAll tables generated successfully.")
print({
    "territories": len(territories), "sales_reps": len(sales_reps), "products": len(products),
    "accounts": len(accounts), "contacts": len(contacts), "leads": len(leads),
    "opportunities": len(opportunities), "pipeline_history": len(pipeline_history),
    "sales_activities": len(sales_activities), "quotes": len(quotes),
    "won_deals": len(won_deals), "lost_deals": len(lost_deals),
})
