//! Options parsed at the language boundary, never compared as strings in IBSS.
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PriorMethod {
    Optim,
    Em,
    Simple,
    None,
}

impl FromStr for PriorMethod {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "optim" => Ok(Self::Optim),
            "EM" => Ok(Self::Em),
            "simple" => Ok(Self::Simple),
            "none" => Ok(Self::None),
            _ => Err(format!(
                "Unsupported prior variance estimation method: {value}"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConvergenceMethod {
    Elbo,
    AlphaPip,
    Joint,
    Pip,
}

impl ConvergenceMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Elbo => "elbo",
            Self::AlphaPip => "alpha_pip",
            Self::Joint => "joint",
            Self::Pip => "pip",
        }
    }
}

impl FromStr for ConvergenceMethod {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "elbo" => Ok(Self::Elbo),
            "alpha_pip" => Ok(Self::AlphaPip),
            "joint" => Ok(Self::Joint),
            "pip" => Ok(Self::Pip),
            _ => Err(format!("Unsupported convergence method: {value}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_options_fail_at_boundary() {
        assert!("uniroot".parse::<PriorMethod>().is_err());
        assert!("unknown".parse::<ConvergenceMethod>().is_err());
    }
}
