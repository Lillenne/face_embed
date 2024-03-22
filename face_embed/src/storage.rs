use s3::{Bucket, Region, creds::Credentials, BucketConfiguration};

pub async fn get_or_create_bucket(bucket_name: &str, endpoint: String, access_key: &str, secret_key: &str) -> anyhow::Result<Bucket> {
    let region = Region::Custom {
        region: "".to_owned(),
        endpoint
    };
    let creds = Credentials::new(Some(access_key), Some(secret_key), None, None, None)?;
    let mut bucket = Bucket::new(bucket_name, region.clone(), creds.clone())?.with_path_style();

    if !bucket.exists().await? {
        bucket = Bucket::create_with_path_style(
            bucket_name,
            region,
            creds,
            BucketConfiguration::default()
        ).await?
        .bucket;
    }

    Ok(bucket)
}
