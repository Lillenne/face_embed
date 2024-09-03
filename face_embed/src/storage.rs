use s3::{creds::Credentials, Bucket, BucketConfiguration, Region};

pub async fn get_or_create_bucket(
    bucket_name: &str,
    endpoint: String,
    access_key: &str,
    secret_key: &str,
) -> anyhow::Result<Bucket> {
    let region = Region::Custom {
        region: "".to_owned(),
        endpoint,
    };
    let creds = Credentials::new(Some(access_key), Some(secret_key), None, None, None)?;

    // Create the bucket
    if let Ok(response) = Bucket::create_with_path_style(
        bucket_name,
        region.clone(),
        creds.clone(),
        BucketConfiguration::default(),
    )
    .await
    {
        if response.response_code == 200 {
            // Created the bucket
            return Ok(*response.bucket);
        }
    }

    // already existed
    let mut bucket = Bucket::new(bucket_name, region, creds)?;
    bucket.set_path_style();
    Ok(*bucket)
}
