//! Immediate local application result for the standalone write path.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApplicationWait {
    Complete,
}
