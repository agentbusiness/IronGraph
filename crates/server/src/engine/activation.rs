//! Standalone artifact activation after local validation and device admission.

use serde::{Deserialize, Serialize};

use crate::graph::EmbeddingProfile;
use crate::{Error, ErrorCode, ProjectId, Result};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationSubject {
    EmbeddingProfile {
        project: ProjectId,
        profile_hash: [u8; 32],
    },
}

impl ActivationSubject {
    fn validate(&self) -> Result<()> {
        match self {
            Self::EmbeddingProfile {
                project,
                profile_hash,
            } if !project.0.is_nil() && *profile_hash != [0; 32] => Ok(()),
            _ => Err(Error::invalid_data("invalid artifact activation subject")),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationBarrier {
    subject: ActivationSubject,
    ready: bool,
}

impl ActivationBarrier {
    pub fn begin(subject: ActivationSubject) -> Result<Self> {
        subject.validate()?;
        Ok(Self {
            subject,
            ready: false,
        })
    }
    #[must_use]
    pub const fn subject(&self) -> &ActivationSubject {
        &self.subject
    }
    pub fn acknowledge(&mut self) -> Result<()> {
        self.validate()?;
        self.ready = true;
        Ok(())
    }
    pub fn finish(self) -> Result<ActivationSubject> {
        self.validate()?;
        if !self.ready {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "artifact activation is not locally ready",
            ));
        }
        Ok(self.subject)
    }
    #[must_use]
    pub const fn is_ready(&self) -> bool {
        self.ready
    }
    pub fn validate(&self) -> Result<()> {
        self.subject.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingProfileActivation {
    profile: EmbeddingProfile,
    barrier: ActivationBarrier,
}

impl EmbeddingProfileActivation {
    pub fn begin(project: ProjectId, profile: EmbeddingProfile) -> Result<Self> {
        profile.validate()?;
        let barrier = ActivationBarrier::begin(ActivationSubject::EmbeddingProfile {
            project,
            profile_hash: profile.profile_hash,
        })?;
        Ok(Self { profile, barrier })
    }
    pub fn acknowledge(&mut self) -> Result<()> {
        self.barrier.acknowledge()
    }
    pub fn finish(self) -> Result<EmbeddingProfile> {
        match self.barrier.finish()? {
            ActivationSubject::EmbeddingProfile { profile_hash, .. }
                if profile_hash == self.profile.profile_hash =>
            {
                Ok(self.profile)
            }
            _ => Err(Error::new(
                ErrorCode::CorruptStorage,
                "embedding profile does not match its readiness barrier",
            )),
        }
    }
    #[must_use]
    pub const fn profile(&self) -> &EmbeddingProfile {
        &self.profile
    }
    #[must_use]
    pub const fn barrier(&self) -> &ActivationBarrier {
        &self.barrier
    }
    pub fn validate(&self) -> Result<()> {
        self.profile.validate()?;
        self.barrier.validate()?;
        match self.barrier.subject() {
            ActivationSubject::EmbeddingProfile { profile_hash, .. }
                if *profile_hash == self.profile.profile_hash =>
            {
                Ok(())
            }
            _ => Err(Error::new(
                ErrorCode::CorruptStorage,
                "embedding profile does not match its readiness barrier",
            )),
        }
    }
}
