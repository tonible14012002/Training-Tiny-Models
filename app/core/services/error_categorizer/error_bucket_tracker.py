import typing as t
from app.core import schemas

class ErrorBucketTracker():
    def __init__(
        self,
        error_buckets: t.List[schemas.ErrorBucket],
        max_examples_per_bucket: int = 10,
        max_generations_per_bucket: int = 3
    ):
        self.error_buckets = {bucket.name: bucket for bucket in error_buckets}
        self.max_examples_per_bucket = max_examples_per_bucket
        self.max_generations_per_bucket = max_generations_per_bucket
        # Track generation count and sample count for each bucket
        self.generation_count = {bucket.name: 0 for bucket in error_buckets}
        self.sample_count = {bucket.name: 0 for bucket in error_buckets}

    def update_bucket(self, bucket_name: str, new_example: schemas.Sample):
        if bucket_name in self.error_buckets:
            self.error_buckets[bucket_name].examples.append(new_example)

    def add_error(self, bucket_name: str, sample: schemas.Sample):
        """
        Add a categorized sample to the correct bucket.
        Increases count for the corresponding bucket and adds to examples list if not exceeding max_examples_per_bucket.

        Args:
            bucket_name: Name of the error bucket
            sample: The sample to add
        """
        if bucket_name not in self.error_buckets:
            return

        # Increase sample count for the bucket
        self.sample_count[bucket_name] += 1

        # Add to examples if under the limit (for few-shot purposes)
        bucket = self.error_buckets[bucket_name]
        if len(bucket.examples) < self.max_examples_per_bucket:
            bucket.examples.append(sample)

    def get_active_buckets(self) -> t.List[schemas.ErrorBucket]:
        """
        Return buckets that are not empty and have not reached the max generation limit.

        Returns:
            List of ErrorBucket objects that are active
        """
        active_buckets = []

        for bucket_name, bucket in self.error_buckets.items():
            # Check if bucket has samples and hasn't reached max generations
            has_samples = self.sample_count[bucket_name] > 0
            under_generation_limit = self.generation_count[bucket_name] < self.max_generations_per_bucket

            if has_samples and under_generation_limit:
                active_buckets.append(bucket)

        return active_buckets