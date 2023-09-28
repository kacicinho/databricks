from databricks_jobs.jobs.ml_models_jobs.pclick.pclick_training_job.entrypoint import PClickTrainingJob


if __name__ == "__main__":
    job = PClickTrainingJob()
    job.inference()
