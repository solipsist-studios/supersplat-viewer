// Creates a self-signed certificate for `npm run serve:https`, once, in .certs/.
//
// WebGPU only exists in a secure context: https://, or http://localhost. Opening
// the dev server from another machine (a phone, a Mac, a headset) over plain http
// falls back to WebGL2, so those tests need https. A self-signed certificate is
// enough; the browser shows a warning that you accept once per origin.
//
// The certificate names localhost, this machine's hostname, and every IPv4 address
// on its network interfaces. Delete .certs/ to regenerate it after the addresses
// change.

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { hostname, networkInterfaces } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = join(dirname(fileURLToPath(import.meta.url)), '..', '.certs');
const key = join(dir, 'key.pem');
const cert = join(dir, 'cert.pem');

if (!existsSync(key) || !existsSync(cert)) {
    const names = new Set(['localhost', hostname()]);
    const addresses = new Set(['127.0.0.1', '::1']);
    for (const list of Object.values(networkInterfaces())) {
        for (const entry of list ?? []) {
            if (entry.family === 'IPv4') {
                addresses.add(entry.address);
            }
        }
    }
    const altNames = [...[...names].map(n => `DNS:${n}`), ...[...addresses].map(a => `IP:${a}`)].join(',');

    mkdirSync(dir, { recursive: true });
    execFileSync('openssl', [
        'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-sha256', '-days', '825',
        '-keyout', key, '-out', cert,
        '-subj', '/CN=supersplat-viewer dev',
        '-addext', `subjectAltName=${altNames}`
    ], { stdio: 'inherit' });
    console.log(`Created ${cert} for ${altNames}`);
}
