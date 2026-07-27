use unitoken::{
  bpe::{BpeTrainer, Idx},
  traits::Encode,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut trainer = BpeTrainer::<u8, Idx>::from_words(
    [("hello", 10), ("world", 7)],
    &[],
  );
  trainer.train_until(260)?;

  let model = trainer.validate_model()?;
  let encoder = model.to_encoder()?;
  let ids = encoder.encode_string("hello")?;

  assert_eq!(encoder.decode(&ids)?, "hello");
  println!("encoded 'hello' as {ids:?}");
  Ok(())
}
