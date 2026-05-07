# =============================================================================
# gpt.py — Full GPT Transformer trained on the Bhagavad Gita
# =============================================================================
# HOW THIS DIFFERS FROM model.py (bigram):
#
# Bigram:      character → lookup one row in a table → predict next char
#              Memory = 0. Sees exactly 1 character.
#
# Transformer: character → look at ALL previous characters → weigh their
#              importance → combine them intelligently → predict next char
#              Memory = block_size characters. Each char "attends" to all others.
#
# The mechanism that does this "looking back and weighing" is called
# SELF-ATTENTION — invented in the 2017 paper "Attention Is All You Need".
# Everything below is building up to that one idea.
# =============================================================================