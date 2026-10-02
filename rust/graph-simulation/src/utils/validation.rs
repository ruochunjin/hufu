//! Synthetic graph data types and generators used to validate solver parity.

use std::{clone, collections::{HashMap, HashSet}, hash::Hash, ops::{Add, BitXor, Div, Mul, Sub}};
use rand::{prelude::*, rng};
use rand::distr::StandardUniform;
use serde::{Serialize, Deserialize};
use std::sync::RwLock;
use rand_pcg::Pcg64;
use lazy_static::lazy_static;

lazy_static!{
    static ref clusters: RwLock<HashMap<u64, Desc>> = RwLock::new(HashMap::new()); 
}

#[derive(Clone, Serialize, Deserialize)]
struct Desc([f64; 16]);

impl Hash for Desc {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for value in &self.0 {
            let bits = value.to_bits();
            bits.hash(state);
        }
    }
}

impl PartialEq for Desc {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(x, y)| x == y)
    }
}

impl Eq for Desc {}

impl Add for Desc {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut result = [0.0; 16];
        for i in 0..16 {
            result[i] = self.0[i] + other.0[i];
        }
        Desc(result)
    }
}

impl Sub for Desc {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let mut result = [0.0; 16];
        for i in 0..16 {
            result[i] = self.0[i] - other.0[i];
        }
        Desc(result)
    }
}

impl Mul for Desc {
    type Output = f64;
    fn mul(self, other: Self) -> f64 {
        let mut result = 0.0;
        for i in 0..16 {
            result += self.0[i] * other.0[i];
        }
        result
    }
}

impl Mul<f64> for Desc {
    type Output = Desc;
    fn mul(self, scalar: f64) -> Desc {
        let mut result = [0.0; 16];
        for i in 0..16 {
            result[i] = self.0[i] * scalar;
        }
        Desc(result)
    }
}

impl Div<f64> for Desc {
    type Output = Desc;
    fn div(self, scalar: f64) -> Desc {
        let mut result = [0.0; 16];
        for i in 0..16 {
            result[i] = self.0[i] / scalar;
        }
        Desc(result)
    }
}

// cosine for Desc
impl BitXor for Desc {
    type Output = f64;
    fn bitxor(self, other: Self) -> f64 {
        let res = self.clone() * other.clone();
        let norm1 = self.clone() * self.clone();
        let norm2 = other.clone() * other.clone();
        res / (norm1.sqrt() * norm2.sqrt())
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone)]
pub struct Node {
    id: u64,
    node_type: u64,
    desc: Desc
}

impl BitXor for Node {
    type Output = f64;
    fn bitxor(self, other: Self) -> f64 {
        return self.desc.clone() ^ other.desc.clone();
    }
}

fn generate_orthogonal_unit(base: &Desc) -> Desc {
    let base_norm = (base.clone() * base.clone()).sqrt();
    let mut orthogonal = Desc([0.0; 16]);
    
    let mut rng = rng();
    loop {
        // Draw a random candidate vector.
        for i in 0..16 {
            orthogonal.0[i] = rng.sample(StandardUniform);
        }
        
        
        // Compute its projection onto the base direction.
        let projection = (orthogonal.clone() * base.clone()) / base_norm;
        
        // Remove the projected component to obtain an orthogonal vector.
        // for i in 0..16 {
        //     orthogonal[i] -= projection * base[i] / base_norm;
        // }

        orthogonal = orthogonal - (base.clone() * projection) / base_norm;
        
        // Normalize only after rejecting numerically degenerate candidates.
        let ortho_norm = (orthogonal.clone() * orthogonal.clone()).sqrt();
        if ortho_norm > 1e-10 {
            orthogonal = orthogonal / ortho_norm;
            break;
        }
    }
    orthogonal
}


impl Node {
    pub fn from_random(id: u64, k: u64, p: f64, alpha: f64) -> Node {
        // get A random [f64; 16]
        let mut rng = rng();
        let random_type = rng.random_range(0..k);
        let desc = {
            let random_vec: [f64; 16] = rng.sample(StandardUniform);
            let desc =    Desc(random_vec);
            
            if !clusters.read().unwrap().contains_key(&random_type) {
                if clusters.read().unwrap().is_empty() {
                    clusters.write().unwrap().insert(random_type, desc.clone());
                    desc
                } else {
                    let avg_vec = clusters.read().unwrap().iter().map(|(_, v)| v.clone()).reduce(|a, b| a + b).unwrap();
                    let orthogonal = generate_orthogonal_unit(&avg_vec);
                    let res = orthogonal + desc;
                    clusters.write().unwrap().insert(random_type, res.clone());
                    res
                }
            } else {
                let cluster_guard = clusters.read().unwrap();
                let cluster_desc = cluster_guard.get(&random_type).unwrap().clone();
    
                if rng.random_bool(p) {
                    let res = cluster_desc.clone() * (1.0 - alpha) + desc * alpha;
                    res
                } else {
                    let orthogonal = generate_orthogonal_unit(&cluster_desc);
                    let res = orthogonal * (1.0 - alpha) + desc * alpha;
                    res
                }
            }
        };

        Node {
            id,
            node_type: random_type,
            desc
        }
    }
}

struct Hyperedge {
    id_set: HashSet<u64>,
}

struct DiHyperedge {
    src: HashSet<u64>,
    dst: HashSet<u64>
}
