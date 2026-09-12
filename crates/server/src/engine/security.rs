//! Durable client credential state for authenticated remote protocol listeners.

use serde::{Deserialize, Serialize};

use crate::Result;

use super::CredentialRegistry;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityState {
    client_credentials: CredentialRegistry,
}

impl SecurityState {
    #[must_use]
    pub const fn client_credentials(&self) -> &CredentialRegistry {
        &self.client_credentials
    }

    pub fn client_credentials_mut(&mut self) -> &mut CredentialRegistry {
        &mut self.client_credentials
    }

    pub fn validate(&self) -> Result<()> {
        self.client_credentials.validate()
    }
}
