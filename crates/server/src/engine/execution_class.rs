//! Local execution-device class used by startup admission.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionClass {
    #[default]
    Cpu,
    Metal,
    Cuda,
}
