const net = require('net');

const port = parseInt(process.argv[2] || '5050', 10);

const server = net.createServer((socket) => {
  socket.setEncoding('utf8');
  let buffer = '';

  socket.on('data', (chunk) => {
    buffer += chunk;
    let idx;
    while ((idx = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;

      let msg;
      try {
        msg = JSON.parse(line);
      } catch (err) {
        socket.write(JSON.stringify({ error: 'invalid_json', details: String(err) }) + '\n');
        continue;
      }

      if (msg.stop) {
        socket.write(JSON.stringify({ stopped: true }) + '\n');
        socket.end();
        server.close();
        return;
      }

      const value = Number(msg.value);
      // console.log(`[node_helper] received value=${value}`);
      const response = {
        logged_by: 'node_socket_helper',
        pid: process.pid,
        value,
        timestamp_ns: BigInt(Date.now()) * 1000000n
      };
      socket.write(JSON.stringify(response, (_, v) => typeof v === 'bigint' ? v.toString() : v) + '\n');
    }
  });
});

server.listen(port, '127.0.0.1', () => {
  console.log(`[node_helper] listening on 127.0.0.1:${port}`);
});
