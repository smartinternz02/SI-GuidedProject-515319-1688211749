import os
from server import app

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        debug=False,
        port=os.environ.get('PORT', 5000)
    )