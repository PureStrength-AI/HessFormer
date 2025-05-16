import hessformer as hf

def test_quick():
    hf.estimate_spectrum(
        model="sshleifer/tiny-gpt2",
        dataset=["hello world"],
        num_iter=2,
        batch_size=1,
        max_length=8,
    )
