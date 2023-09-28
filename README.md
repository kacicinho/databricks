# databricks-jobs

This project regroups the main batch job used to do : 
- personalised recommendation jobs
- building some fact tables
- training ML models
- exporting data 
- creating metadata

## Structure 
The jobs are grouped in folders based on the main intent : 
- `fact_table_build_jobs` : table like fact_audience are built there
- `metadata_enrichment_jobs` : used to compute highlevel infos like tags from synopsis or audio
- `ml_models_jobs` : jobs there train or execute inference from ml models
- `personalised_reco_jobs` : home of the this_week, this_day and personality recommendations used in the product
- `program_similarity_jobs` : program similarity jobs (currently used for Mango only)
- `third_party_export_jobs` : these jobs usually build from user information and export to an external partner like Liveramp
- `misc_jobs` : these jobs are more utils like the S3 data to AWS RDS job
- `utils` : does not contain jobs, only function reused in other jobs

## Installing project requirements

```bash
 brew cask install java
 brew install spark
```

## Install project package in a developer mode
It is recommended to build a virtual env before installing anyting (python3 -m venv venv)
```bash
pip install -r unit-requirements.txt
pip install -e .
```

## Configure your databricks setup

You need to have your `~/.databrickscfg` file setup to deploy your code
```
echo "[DEFAULT]" >> ~/.databrickscfg
echo "host = https://mtv-data-dev.cloud.databricks.com" >> ~/.databrickscfg
echo "token = *********" >> ~/.databrickscfg
```

## Install dbx

cd tools
pip install dbx-0.7.0-py3-none-any.whl


## Testing

For local unit testing, please use `pytest`:
```
pytest tests/unit
```

Check the code style with 
```
 flake8 ./databricks_jobs --count --select=E9,F63,F7,F82 --show-source --statistics &&
 flake8 ./databricks_jobs --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

Deploy your code, a few examples
```
## Prefered way
dbx deploy --jobs=databricks_jobs-popular_reco_job && dbx launch --job=databricks_jobs-popular_reco_job
## Alternate ways
dbx deploy # It will deploy all the jobs in the repo
dbx launch --job=databricks_jobs-popular_reco_job --trace 
```

## Common errors & fix in local testing
```
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES / for multiprocessing issue
export JAVA_HOME=$(/usr/libexec/java_home -v 1.8) / Use Java 8 for debugging. pyarrow pandas_udf easily compatible with java 8
```

## Deployment on Github actions 

Defined in .github/workflows
On master merge, code is deployed on the production env.

If the yaml file must be modified, one can use `act` : 
```
 act -e /Users/amorvan/Documents/code_dw/databricks_dev_repo/databricks_jobs/.github/workflows/deploy_jobs_on_prod_databricks.yml -s DATABRICKS_HOST=https://mtv-data-dev.cloud.databricks.com -s DATABRICKS_TOKEN=XXXXXXX -s SNOWFLAKE_USER=XXXXXXXX -s SNOWFLAKE_PASSWORD=XXXXXX  -s AWS_ONBOARDING_ACCESS_KEY=XXXXXXX -s AWS_ONBOARDING_SECRET_KEY=XXXXXXX -s AWS_S3_ONBOARDING_OUTPUT_BUCKET=mtv-dev-onboarding -s AWS_S3_PANEL_OUTPUT_BUCKET=mtv-dev-panel --verbose
```

Act is also used to run pull_request tests as in Github Actions, run this command to run it locally.
```
act pull_request
```
Note : you may have errors arounf spacy model downloads, they usually don't happen on the CI.

## CICD pipeline settings and secrets

In order to function, the deployment script needs secrets like :
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
These values are stored in the Github interface for secrets : https://github.com/MolotovTv/databricks-jobs/settings/secrets/actions
  
The secrets added here are supposed to be strictly necessary for the correct deployment of jobs.
We now try to avoid adding more secrets in this place.
  
The common way of adding a new secret in a job is through the parameter store : https://docs.aws.amazon.com/fr_fr/systems-manager/latest/userguide/systems-manager-parameter-store.html
This means that the secrets will not be provided in the config.json of the job but rather retrieved through an API call.


# Sanity checks on recos 

**General idea** : when a recommendation output seems suspicious, you can use one of these queries to diagnose what the metrics used by the jobs output.
Doing so allows to quickly see if the anomaly is in the data or in the job logic.


For bookmarks 

```sql
select TITLE, count(distinct USER_ID) as cnt
from backend.scheduled_recording 
join backend.program
on backend.program.id = backend.scheduled_recording.program_id
where backend.scheduled_recording.SCHEDULED_AT > current_date - 14
group by TITLE
order by cnt desc
```

For allo ciné

```sql 
with  free_channel as (
 select CHANNEL_ID, TVBUNDLE_ID, bc.NAME, bc.DISPLAY_NAME 
            from backend.rel_tvbundle_channel
            inner join backend.channel as bc on CHANNEL_ID = bc.ID
            where tvbundle_id in (25, 90, 26, 31, 60)
 )
select distinct AFFINITY, backend.program.TITLE, rating
from backend.program_rating
join backend.broadcast
on backend.program_rating.PROGRAM_ID = backend.broadcast.PROGRAM_ID
join backend.program 
on backend.program.ID = backend.broadcast.PROGRAM_ID
left join external_sources.daily_prog_affinity
on backend.program.ID=external_sources.daily_prog_affinity.program_id
join free_channel
on free_channel.CHANNEL_ID = backend.broadcast.CHANNEL_ID
where REF_PROGRAM_CATEGORY_ID = 1 and PROGRAM_RATING_SERVICES_ID = 2 and PROGRAM_RATING_TYPE_ID = 2
and START_AT >= current_date and START_AT <= current_date + 7
order by rating desc
```

For celeb_points

```sql 
with temp as (
  select PERSON_ID, FIRST_NAME, LAST_NAME, count(*) as cnt 
  from backend.person
  left outer join backend.user_follow_person 
  on PERSON_ID = ID
  where source = 'molotov'
  group by PERSON_ID, FIRST_NAME, LAST_NAME
),
 rel_person_prog as (
    select PROGRAM_ID, PERSON_ID
    from backend.rel_program_person
    UNION (
      select program_id, person_id 
      from backend.rel_episode_person 
      join backend.episode 
      on episode_id = id 
     )
),
pop_score_per_program as (
  select p.PROGRAM_ID, SUM(coalesce(temp.cnt, 0)) as total_celeb_points
    from temp
    join rel_person_prog as p
    on p.PERSON_ID = temp.PERSON_ID
  group by p.PROGRAM_ID
),
 free_channel as (
 select CHANNEL_ID, TVBUNDLE_ID, bc.NAME, bc.DISPLAY_NAME 
            from backend.rel_tvbundle_channel
            inner join backend.channel as bc on CHANNEL_ID = bc.ID
            where tvbundle_id in (25, 90, 26, 31, 60)
 )

select distinct AFFINITY, pop_score_per_program.PROGRAM_ID, bp.TITLE, total_celeb_points
from pop_score_per_program
join backend.program as bp
on ID = pop_score_per_program.PROGRAM_ID
left join external_sources.daily_prog_affinity
on bp.id=external_sources.daily_prog_affinity.program_id
join backend.broadcast as bb
on bb.PROGRAM_ID = bp.ID
join free_channel as fc
on bb.CHANNEL_ID = fc.CHANNEL_ID
where REF_PROGRAM_CATEGORY_ID = 1 and START_AT >= current_date and START_AT < current_date + 7
order by total_celeb_points desc
```

Replay metrics : 
```
with cte_total_watch as (
    select PROGRAM_ID, sum(DURATION) as total_watch
    from dw.fact_watch
    where date_day > current_date() - 7
    group by PROGRAM_ID
),
cte_total_replay_watch as (
    select PROGRAM_ID, sum(DURATION) as total_replay_watch
    from dw.fact_watch
    where date_day > current_date() - 7 and asset_type = 'replay'
    group by PROGRAM_ID
),
cte_total_people_replay as (
    select PROGRAM_ID, count(distinct USER_ID) as total_people_replay
    from dw.fact_watch
    where date_day > current_date() - 7 and asset_type = 'replay' and DURATION > 30
    group by PROGRAM_ID
),
cte_available_replay as (
    SELECT
    DISTINCT p.id
    , p.title
    , rtc.tvbundle_id
    , t.name
    FROM backend.vod v
        INNER JOIN backend.episode e ON v.episode_id = e.id
        INNER JOIN backend.program p ON e.program_id = p.id
        INNER JOIN backend.channel c ON v.channel_id = c.id
        INNER JOIN backend.rel_tvbundle_channel rtc ON v.channel_id = rtc.channel_id
        INNER JOIN backend.tvbundle t ON rtc.tvbundle_id = t.id    
    WHERE v.disabled = 0 

    AND t.is_commercialized = 1
    AND v.available_from <= current_date()
    AND v.available_until >= current_date()
    AND v.withdrawn_at IS NULL
    and v.deleted_at IS NULL
)
select distinct ID, TITLE, total_watch, total_replay_watch, total_people_replay
from cte_available_replay
join cte_total_watch
ON cte_total_watch.PROGRAM_ID = cte_available_replay.ID
join cte_total_replay_watch
ON cte_total_replay_watch.PROGRAM_ID = cte_available_replay.ID
join cte_total_people_replay
ON cte_total_people_replay.PROGRAM_ID = cte_available_replay.ID
order by total_people_replay desc
```

Top replay verification query (approxmate)
```sql
with cte_broadcast_watch as (
  select backend.program.title, episode_id, 
    sum(fw.duration) as total_duration, 
    count(CASE WHEN fw.duration > GREATEST(20 * 60, 0.8 * backend.episode.duration) THEN USER_ID ELSE NULL END) as nb_engaged_watchers,
    min(real_start_at) as first_aired_at
  from dw.fact_watch as fw
  join backend.program 
  on id = program_id
  join backend.episode
  on episode_id = backend.episode.id
  where real_start_at > current_date - 7 and fw.DURATION > 15 * 60
  group by backend.program.title, episode_id
  order by total_duration desc
),
cte_available_replay as (
    SELECT
    DISTINCT p.id, e.id as episode_id
    FROM backend.vod v
        INNER JOIN backend.episode e ON v.episode_id = e.id
        INNER JOIN backend.program p ON e.program_id = p.id
        INNER JOIN backend.channel c ON v.channel_id = c.id
        INNER JOIN backend.rel_tvbundle_channel rtc ON v.channel_id = rtc.channel_id
        INNER JOIN backend.tvbundle t ON rtc.tvbundle_id = t.id    
    WHERE v.disabled = 0 
    AND v.video_type = 'REPLAY' 
    AND t.is_commercialized = 1
    AND v.available_from <= current_date()
    AND v.available_until >= current_date()
    AND v.withdrawn_at IS NULL
    and v.deleted_at IS NULL
),
cte_reco as (
with cte as (
  select USER_ID, value:EPISODE_ID as EPISODE_ID, value:PROGRAM_ID as rec_prog_id, value:reco_origin as origin, value:rating as rating
  from ML.USER_REPLAY_RECOMMENDATIONS_LATEST, lateral flatten(input => parse_json(recommendations))
  where USER_ID in (0) and UPDATE_DATE = current_date - 1
)
select EPISODE_ID, p.title, origin, rating, CASE WHEN REF_PROGRAM_CATEGORY_ID in (1, 2, 8) THEN 7 else 3 END as max_age
from cte
join backend.program p
on p.ID = rec_prog_id
order by rating desc
)
select backend.program.title, rating,
       round(total_duration / 3600 * (1 - 0.5 * (datediff('DAY', first_aired_at, current_date) / max_age)) * 1, 0 ) as weighted_total_watch, 
       round(nb_engaged_watchers * (1 - 0.5 * (datediff('DAY', first_aired_at, current_date) / max_age)) * 1, 0 ) as weighted_engaged_watchers ,
       first_aired_at, datediff('DAY', first_aired_at, current_date) as age, ar.episode_id
from cte_broadcast_watch as bw
join cte_available_replay as ar
on ar.episode_id = bw.episode_id
join backend.program 
on ar.id = backend.program.id
left join cte_reco
on cte_reco.EPISODE_ID = bw.EPISODE_ID
where not (REF_PROGRAM_KIND_ID in (36, 43) and REF_PROGRAM_CATEGORY_ID = 4) and age <= max_age 
--qualify row_number() over(partition by backend.program.title order by total_duration desc) = 1
order by weighted_engaged_watchers desc
```