import argparse

class TestOptions():
    def __init__(self):
        parser = argparse.ArgumentParser(description = 'Options for testing')
        parser.add_argument("--test_image_folder", required = True)
        parser.add_argument("--generated_image_folder", required = True)
        parser.add_argument("--generator_type", type = str, default = "mask", help = '[mask, x_ray]')

        self.parser = parser

    def gather_options(self):
        return self.parser.parse_args()
