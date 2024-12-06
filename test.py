import ast

data = f"""['Almost all living organisms, from humans to bacteria, possess a circadian clock. This internal timekeeping system enables organisms to anticipate and adapt to the rhythmic environmental changes that occur over a 24-hour cycle. At their molecular core, these clocks are comprised of cell-autonomous transcription-translation negative feedback loops.', 'In most animals, the master circadian clock in the brain receives light cues via the eyes, which enable synchronization with the external 24-hour light-dark cycles. The master clock sits at the top of the hierarchy and, in turn, modulates the activity of downstream neurons, as well as the peripheral clocks located in tissues throughout the body via endocrine and systemic signaling. In vertebrates, the master clock is located in the suprachiasmatic nucleus of the hypothalamus and is comprised of approximately 20,000 neurons.']"""

data = ast.literal_eval(data)
for d in data:
    print(d)
