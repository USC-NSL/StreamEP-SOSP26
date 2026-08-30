import argparse

def _serve(args, extra_argv):
    from sglang.cli.serve import serve

    return serve(args, extra_argv)


def _generate(args, extra_argv):
    from sglang.cli.generate import generate

    return generate(args, extra_argv)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the SGLang server.",
        add_help=False,  # Defer help to the specific parser
    )
    serve_parser.set_defaults(func=_serve)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run inference on a multimodal model.",
        add_help=False,  # Defer help to the specific parser
    )
    generate_parser.set_defaults(func=_generate)

    args, extra_argv = parser.parse_known_args()
    args.func(args, extra_argv)
