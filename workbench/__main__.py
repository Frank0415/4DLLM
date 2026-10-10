import argparse
import threading
import webbrowser

import uvicorn


def main():
    parser = argparse.ArgumentParser(description='Local 4DLLM browser workbench')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not args.no_browser:
        timer = threading.Timer(1.5, webbrowser.open, args=(f'http://127.0.0.1:{args.port}',))
        timer.daemon = True
        timer.start()
    uvicorn.run('workbench.app:app', host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
