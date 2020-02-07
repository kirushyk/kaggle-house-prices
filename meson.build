project('kaggle-house-prices', 'c',
    version: '0.0.1',
    meson_version: '>= 0.49.0',
    default_options: [
        'c_std=c99',
        'warning_level=1',
        'buildtype=debug'
    ]
)

learn_prices_sources = [
    'main.c'
]

le = dependency('le')

incdir = include_directories('.')

executable('learn-prices', learn_prices_sources,
    dependencies: le,
    include_directories: incdir,
    install: false
)
