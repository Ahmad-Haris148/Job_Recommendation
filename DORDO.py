# DO for Job
class JobDO:
    def _init(self, job_id: int, title: str, skills_required: list): # Changed _init to _init_
        self.job_id = job_id
        self.title = title
        self.skills_required = skills_required

    def _repr(self): # Changed _repr to _repr_
        return f"JobDO(job_id={self.job_id}, title='{self.title}', skills_required={self.skills_required})"


# DO for User
class UserDO:
    def _init(self, user_id: int, name: str, skills: list): # Changed _init to _init_
        self.user_id = user_id
        self.name = name
        self.skills = skills

    def _repr(self): # Changed _repr to _repr_
        return f"UserDO(user_id={self.user_id}, name='{self.name}', skills={self.skills})"


# RDO for Job
class JobRepository:
    def _init(self): # Changed _init to _init_
        self.jobs = []  # In-memory storage for jobs

    def add_job(self, job: JobDO):
        self.jobs.append(job)
        return job

    def get_job_by_id(self, job_id: int):
        return next((job for job in self.jobs if job.job_id == job_id), None)

    def get_all_jobs(self):
        return self.jobs


# RDO for User
class UserRepository:
    def _init(self): # Changed _init to _init_
        self.users = []  # In-memory storage for users

    def add_user(self, user: UserDO):
        self.users.append(user)
        return user

    def get_user_by_id(self, user_id: int):
        return next((user for user in self.users if user.user_id == user_id), None)

    def get_all_users(self):
        return self.users


# Business Logic for Job Recommendations
class JobRecommendationService:
    def _init(self, user_repo: UserRepository, job_repo: JobRepository): # Changed _init to _init_
        self.user_repo = user_repo
        self.job_repo = job_repo

    def recommend_jobs(self, user_id: int):
        user = self.user_repo.get_user_by_id(user_id)
        if not user:
            return f"User with ID {user_id} not found."

        recommended_jobs = []
        user_skills = set(user.skills)

        for job in self.job_repo.get_all_jobs():
            job_skills = set(job.skills_required)
            if job_skills.issubset(user_skills):
                recommended_jobs.append(job)

        return recommended_jobs

# Main Function to Test the Code
if __name__== "_main": # Changed _name to _name_ and "main" to "_main_"
    # Initialize repositories
    user_repo = UserRepository()
    job_repo = JobRepository()

    # Add sample jobs
    job1 = JobDO(job_id=1, title="Software Engineer", skills_required=["Python", "Django"])
    job2 = JobDO(job_id=2, title="Data Scientist", skills_required=["Python", "Machine Learning", "Pandas"])
    job_repo.add_job(job1)
    job_repo.add_job(job2)

    # Add a sample user
    user1 = UserDO(user_id=1, name="Alice", skills=["Python", "Django", "Machine Learning"])
    user_repo.add_user(user1)