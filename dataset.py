import tiktoken

def get_input_and_target(corpus_path: str) -> tuple[list[int], list[int]]:
    tok = tiktoken.get_encoding(encoding_name='gpt2')

    eot = '<|endoftext|>'
    eot_id = tok.encode(eot, allowed_special={'<|endoftext|>'})

    input = eot_id.copy()
    target = []
    with open(corpus_path) as r_h:
        for line in r_h.readlines():
            if line:
                tokenised_line = tok.encode(line)
                input.extend(tokenised_line + eot_id)
                target.extend(tokenised_line + eot_id)

    return input, target

                  
if __name__ == "__main__":
    ds = get_input_and_target("/home/piotrz/git/GPT2-Small/GPT2-Small/input.txt")
    print(type(ds))