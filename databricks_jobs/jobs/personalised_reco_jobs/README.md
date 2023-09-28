## Personalised jobs

#### General philosophy
These jobs mostly follow the same logic in order to do recommendation : 

- Compute a filtered set of eligible programs
- order them based on metrics
- Broadcast to all users
- Filter out programs based on user bundles and settings
- Re-order based on user profile (OPTIONNAL)
- Compute default recommendations (user 0 and negative id, default bundle recommendation)


#### Drill down on this_week
The most advanced job with the described philosophy is this_week aka `popular_reco_job`.
Here we will detail all the steps.

```
1 - Load available programs (all broadcasts in the next 7 days on usual Molotov bundles)
2 - Stitch popularity infos on ech prog
3 - Compute top K infos based on popularity and affinity
4 - Broadcast recos on user with enough information
5 - Filter to keep only N recos
    * The bundle available to the user is used to filter programs
    * a formula based on rank of programs is used (specific part of this_week)
    * several rules are used to avoid extrem results (again specific, eg: banned kinds, max nb result per affinity)
6 - Complete with generic recommendation
    * the basic reco is based on bundles => build_best_of_bundle
    * the user 0 recommendation is used to provide something for inactive or unknown users
      => build_non_personalised_reco
7 - Store results
8 - Similar to user 0 other fake users are added bundle wise
```

#### Building your own logic

The steps presented in the previous part are very generic. 
Some parts here and there are specific to the this_week logic. In fact, the top_replay is extremely similar.
In general, it almost possible to copy paste the whole logic with different metrics with low effort.

Let's see how we could do it.


**My_new_very_impressive_algo**
```
1 - TODO : define the new datasource
2 - join_pop_info_to_programs(ouput_of_1_df, my_metrics).persist()
3 - keep_top_X_programs(ouput_of_2_df, my_metrics, **other_params)
4 - TODO : Broadcast recos on users 
    - Example : select_recos in reco_replay_job - crossJoin is used from fact_audience users
5 - select_among_categories(ouput_of_4_df, 
                            allowed_channels_per_user, 
                            allowed_csa_per_user, **params)
6 - User 0 is mandatory to have a default recommendation :
    - build_best_of_bundle provide a generic way to bbuild these default recommendation
7 - Store results
8 - OPTIONAL
```

#### Closing thoughts
In the simplest form of algorithms, only a user 0 recommendation would exist.

From there, one could make the algorithm more and more complex by adding personalisation; first by bundle, 
then by taking into account the watch history and finaly with a custom personalisation logic (affinity watch in the case of this_week).


**What could be the next jobs with this logic ?**
- VOD recommendation
- To some extend : a "because you watched X" rail, where one would need to know which programs would be used as queries for the usage of a program similarity algorithm