import pandas as pd
df = pd.read_csv(r"P:\Data Analytics\Projects\3\raw_data.csv")

import ast

# Convert token strings into list format for analysis
df["description_tokens"] = df["description_tokens"].apply(lambda x : ast.literal_eval(x) if isinstance(x, str) else [])

# Check dataset size
print(df.shape)

# Review columns
print(df.columns)

# Verify data types
print(df.dtypes)

# Check missing values
print(df.isnull().sum().sort_values(ascending = False))

# Inspect sample data
print(df.head(5))
print(df.sample(5, random_state = 42))

# Remove unnecessary columns
cols_to_drop = ["Unnamed: 0", "index", "commute_time", "thumbnail"]
df = df.drop(columns = cols_to_drop, errors='ignore')

# Normalize job titles
df["title_clean"] = df["title"].str.lower()

# Map job titles to role categories
def map_role(title):
    if "data analyst" in title:
        return "Data Analyst"
    elif "business analyst" in title:
        return "Business Analyst"
    elif "bi analyst" in title:
        return "BI Analyst"
    elif "data scientist" in title:
        return "Data Scientist"
    else:
        return "Other"

df["role"] = df["title_clean"].apply(map_role)

# Flag salary availability
df["salary_disclosed"] = df["salary_standardized"].notna()

# Identify remote roles
df["is_remote"] = (
    df["location"].str.contains("Anywhere", case = False, na = False)
    | (df["work_from_home"] == True)
    | df["extensions"].str.contains("work_from_home", case = False, na = False)
)

# Clean location values
df["location"] = df["location"].fillna("Unknown").str.strip()

# Count skills per job
df["skill_count"] = df["description_tokens"].apply(len)

# Validate cleaned dataset
print(df.shape)
print(df.head())
print(df.isnull().sum().sort_values(ascending = False))

# Collect all skills
from collections import Counter
all_skills = df["description_tokens"].explode()

# Remove null or empty values
all_skills = all_skills.dropna()
all_skills = all_skills[all_skills != ""]

# Count skill frequency
skill_counts = Counter(all_skills)

# Convert to DataFrame
skill_df = (
    pd.DataFrame(skill_counts.items(), columns = ["Skill", "Count"])
    .sort_values("Count", ascending =False)
)

skill_df.to_csv(r"P:\Data Analytics\Projects\3\skill_df.csv", index = False)

# View top skills
print(skill_df.head(15))

# Calculate demand percentage
total_jobs = df.shape[0]
skill_df["Demand_%"] = (skill_df["Count"]/total_jobs)*100

# Visualize top skills
import matplotlib.pyplot as plt
top_skills = skill_df.head(10)

plt.figure(figsize = (10,5))
plt.barh(top_skills["Skill"], top_skills["Count"])
plt.gca().invert_yaxis()
plt.title("Top 10 In Demand Skills for Data Analyst Roles")
plt.xlabel("Number of Job Postings")
plt.show()

# Analyze skill combinations
from itertools import combinations
skill_pairs = []
for skills in df["description_tokens"] :
    if len(skills)>1:
        skill_pairs.extend(combinations(set(skills), 2))

pair_counts = Counter(skill_pairs)

pair_df = (pd.DataFrame(pair_counts.items(), columns =["Skill_Pair", "Count"])).sort_values("Count", ascending = False)

print(pair_df.head(10))

# Compare remote vs on-site jobs
remote_counts = df["is_remote"].value_counts().reset_index()
remote_counts.columns = ["Remote_Status", "Job_Count"]
print(remote_counts)

# Visualize remote vs on-site
remote_counts.plot(
    x = "Remote_Status",
    y = "Job_Count",
    kind = "bar",
    legend = False
)
plt.title("Remote vs On-Site Data Analyst Jobs")
plt.xlabel("Job Type")
plt.ylabel("Number of Job Postings")
plt.show()

# Top hiring locations
top_locations = (df["location"].value_counts().head(10).reset_index())
top_locations.columns = ["Location", "Job_Count"]
print(top_locations)

# Visualize locations
plt.figure(figsize = (10, 5))
plt.barh(top_locations["Location"], top_locations["Job_Count"])
plt.gca().invert_yaxis()
plt.title("Top 10 Hiring Locations for Data Analyst Roles")
plt.xlabel("Number of Job Postings")
plt.tight_layout()
plt.show()

# Remote jobs by location
remote_by_location = (df[df["is_remote"]== True]["location"].value_counts().head(10).reset_index())
remote_by_location.columns = ["Location", "Remote_Job_Count"]
print(remote_by_location)

# Salary disclosure distribution
salary_counts = (df["salary_disclosed"].value_counts().reset_index())
salary_counts.columns = ["Salary_Disclosed", "Job_Count"]
print(salary_counts)

# Visualize salary disclosure
salary_counts.plot(
    x = "Salary_Disclosed",
    y = "Job_Count",
    kind = "bar",
    legend = False
)
plt.title("Salary Disclosure in Data Analyst Job Postings")
plt.xlabel("Salary Disclosed")
plt.ylabel("Number of Job Postings")
plt.show()

# Salary disclosure by role
salary_by_role = (
    df.groupby("role")["salary_disclosed"]
    .mean()
    .sort_values(ascending = False)
    .reset_index()
)
salary_by_role["salary_disclosed_pct"] = salary_by_role["salary_disclosed"] * 100
print(salary_by_role)

# Salary disclosure by remote status
salary_remote = (
    df.groupby("is_remote")["salary_disclosed"]
    .mean()
    .reset_index()
)
salary_remote["salary_disclosed_pct"] = salary_remote["salary_disclosed"] * 100
print(salary_remote)

# Median salary by role (where available)
median_salary = (
    df[df["salary_disclosed"] == True]
    .groupby("role")["salary_standardized"]
    .median()
    .sort_values(ascending = False)
    .reset_index()
)
print(median_salary)

# Save cleaned dataset at final stage
df.to_csv(r"P:\Data Analytics\Projects\3\clean_data.csv", index = False)
