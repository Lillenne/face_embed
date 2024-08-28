use std::path::Path;

use anyhow::bail;

pub fn expand_path(path: &str) -> anyhow::Result<String> {
    let path = Path::new(path);
    if path.is_relative() {
        let cwd = std::env::current_dir()?;
        let mut bp = std::path::PathBuf::new();
        bp.push(cwd);
        bp.push(path.as_os_str());
        if let Ok(path) = std::fs::canonicalize(bp) {
            if let Some(str) = path.as_os_str().to_str() {
                Ok(str.into())
            } else {
                bail!("Failed to get relative path: {:?}", path)
            }
        } else {
            bail!("Failed to get relative path")
        }
    } else if let Some(str) = path.to_str() {
        Ok(str.to_string())
    } else {
        bail!("Path contains invalid characters")
    }
}

pub fn path_parser(path: &str) -> anyhow::Result<String> {
    let path = expand_path(path)?;
    if Path::new(path.as_str()).exists() {
        Ok(path)
    } else {
        bail!("Path does not exist!")
    }
}
