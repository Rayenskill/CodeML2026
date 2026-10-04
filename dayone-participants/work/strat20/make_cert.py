"""Self-signed TLS certificate for the edge box, so phones on the facility Wi-Fi open the app over HTTPS.

Browsers only give WebCrypto (the phone's encryption) to secure origins: https://… or http://localhost. With this
self-signed certificate the phone shows a warning once ("continue to <box-ip>") and everything works while the box
is reachable — but browsers refuse to register a service worker on an untrusted certificate, so the app cannot be
*reopened* away from the box. For real offline use, install a locally trusted certificate: `mkcert -install`, then
`mkcert -cert-file box_tls.crt -key-file box_tls.key <box-ip> localhost` in the state folder, and install mkcert's
root CA on each phone (once). The certificate is regenerated when the box's addresses change.

python make_cert.py <out_dir>   -> <out_dir>/box_tls.crt, <out_dir>/box_tls.key (valid 2 years, all local IPs)
"""
import datetime as dt
import ipaddress
import socket
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


def local_ips():
    ips = {"127.0.0.1"}
    try:
        ips.update(i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    try:                                    # the address used to reach the LAN (no packet is sent)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ips)


def main(out_dir):
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    crt, key_f = out / "box_tls.crt", out / "box_tls.key"
    if crt.exists() and key_f.exists():               # reuse it while it still names every current address
        old = x509.load_pem_x509_certificate(crt.read_bytes())
        if old.issuer != old.subject or "DayOne edge box" not in old.subject.rfc4514_string():
            print(f"{crt}: user-provided certificate (e.g. mkcert), left untouched")
            return crt, key_f
        names = {str(ip) for ip in old.extensions.get_extension_for_class(x509.SubjectAlternativeName)
                 .value.get_values_for_type(x509.IPAddress)}
        if set(local_ips()) <= names and old.not_valid_after_utc > dt.datetime.now(dt.timezone.utc):
            print(f"{crt} is valid for {', '.join(sorted(names))}")
            return crt, key_f
        print("box address or validity changed: new certificate (phones must accept it again)")
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "DayOne edge box")])
    now = dt.datetime.now(dt.timezone.utc)
    san = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in local_ips()]
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=730))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    key_f.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                        serialization.NoEncryption()))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"wrote {crt} for {', '.join(local_ips())}")
    return crt, key_f


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "dayone_local" / "edge_state"))
