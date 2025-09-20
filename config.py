API_KEYS = {
    'alpha_vantage': 'GR78OL1G1PAPSRG0',
    'twelve_data': '37df463d6a014ea0bb7f6b96558cfb9f',
}

# Model Configuration (Mac M2 Optimized)
MODEL_CONFIG = {
    'sequence_length': 60,
    'batch_size': 16,
    'learning_rate': 0.001,
    'epochs': 100,
    'device': 'mps',
    'mixed_precision': True,
}
