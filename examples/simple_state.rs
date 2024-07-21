use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::{extension::BinomialExtensionField, Field, PrimeField32};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{prove, verify, StarkConfig};
use rand::{distributions::{Distribution, Standard}, thread_rng, Rng};
use tracing_forest::{util::LevelFilter, ForestLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};

const SS_ROW_WIDTH: usize = 3;

struct SimpleState {}

impl<F> BaseAir<F> for SimpleState {
    fn width(&self) -> usize {
        SS_ROW_WIDTH
    }
}

impl<AB: AirBuilder> Air<AB> for SimpleState {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &SimStateRow<AB::Var> = (*local).borrow();
        let next: &SimStateRow<AB::Var> = (*next).borrow();

        builder.when_transition().assert_eq(local.balance + local.input - local.output, next.balance);
    }
}

// this enables both `Var` and `Val` 
struct SimStateRow<F> {
    pub balance: F,
    pub input: F,
    pub output: F
}

impl<F> SimStateRow<F> {
    
}

impl<F> Borrow<SimStateRow<F>> for [F] {
    fn borrow(&self) -> &SimStateRow<F> {
        debug_assert_eq!(self.len(), SS_ROW_WIDTH);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<SimStateRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

// fn generate_next_ss_row<F: PrimeField32>(cur_row: &SimStateRow<F>, next_input: F, next_output: F) -> SimStateRow<F> {
//     let next_balance = cur_row.balance + cur_row.input - cur_row.output;
//     debug_assert!(next_balance + next_input >= next_output, "invalid transaction");

//     SimStateRow { balance: next_balance, input: next_input, output: next_output }
// }

fn random_trace<F: PrimeField32>() -> RowMajorMatrix<F> where Standard: Distribution<F> {
    let n = 1024;
    let mut trace = RowMajorMatrix::new(vec![F::zero(); n * SS_ROW_WIDTH], SS_ROW_WIDTH);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<SimStateRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows[0] = SimStateRow {
        balance: F::from_canonical_u32(100000),
        input: F::from_canonical_u32(12345),
        output: F::from_canonical_u32(54321)
    };

    let mut rng = thread_rng();
    for i in 1..rows.len() {
        let last_row_i = i - 1;
        let next_balance = rows[last_row_i].balance + rows[last_row_i].input - rows[last_row_i].output;
        let next_input: F = rng.gen();
        let high = next_balance.as_canonical_u32() + next_input.as_canonical_u32();
        let low = high * 2 / 3;
        let next_output = F::from_canonical_u32(rng.gen_range(low..high));

        rows[i] = SimStateRow {
            balance: next_balance,
            input: next_input,
            output: next_output
        };
    }

    trace
}

fn main() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();
    
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut thread_rng(),
    );

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs = FieldMerkleTreeMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        MyHash,
        MyCompress,
        8,
    >;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

    let fri_config = FriConfig {
        log_blowup: 2,
        num_queries: 40,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_config);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);

    let trace = random_trace::<Val>();

    let mut p_challenger = Challenger::new(perm.clone());
    let proof = prove(&config, &SimpleState {}, &mut p_challenger, trace, &vec![]);
    let mut v_challenger = Challenger::new(perm);
    verify(&config, &SimpleState {}, &mut v_challenger, &proof, &vec![]).unwrap();
}