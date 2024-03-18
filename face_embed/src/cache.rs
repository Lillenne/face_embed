use std::{time::Duration, time::Instant};
use num_traits::Float;

use crate::embedding::similarity;

pub trait EmbeddingRef<T: Float> {
    fn embedding_ref(&self) -> &[T];
}

pub struct SlidingVectorCache<T: Float, V: EmbeddingRef<T> + Clone> {
    cache: Vec<(Instant, V)>,
    duration: Duration,
    threshold: T
}

impl<T: Float, V: EmbeddingRef<T> + Clone> SlidingVectorCache<T, V> {
    pub fn new(duration: Duration, threshold: T) -> Self {
        let cache: Vec<(Instant, V)> = Vec::new();
        Self { cache, duration, threshold }
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn push(&mut self, time: Instant, vec: &V) -> bool {
        // purge outdated cache members
        self.cache.retain(|(time, _)| std::time::Instant::now() - *time < self.duration);
        let embedding_ref = vec.embedding_ref();
        for (_,v) in self.cache.iter() {
            let sim = similarity(v.embedding_ref(), embedding_ref);
            if sim > self.threshold { return false }
        }
        self.cache.push((time, vec.clone()));
        true
    }
}
