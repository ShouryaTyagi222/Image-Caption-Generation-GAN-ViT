

class Config:
    def __init__(self) -> None:
        self.max_len=35  # Output dimension of the generator (size of vocabulary)
        self.epochs=10
        self.batch_size=32
        self.vocab_size=3000
        self.max_caption_length=20
        self.train_gen_steps=4