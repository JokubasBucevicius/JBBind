# Running the app as a service

`make serve` dies with the terminal. If the browser is on another machine the SSH
tunnel usually outlives it, and a forward with nothing behind it does not fail —
it accepts the connection and holds it, so the tab spins on "loading" forever.
A user service avoids that by simply always being up.

```bash
cp deploy/jbbind.service ~/.config/systemd/user/jbbind.service
systemctl --user daemon-reload
systemctl --user enable --now jbbind
loginctl enable-linger "$USER"      # survive logout and reboot
```

`%h` expands to the home directory, but the voronota and interpreter paths in the
unit are still this installation's — check them before copying it elsewhere.

```bash
systemctl --user status jbbind         # is it up
journalctl --user -u jbbind -f         # what it is doing
systemctl --user restart jbbind        # after a code change (there is no --reload)
```

The port stays on loopback. Reach it with `ssh -N -L 8000:127.0.0.1:8000 user@host`,
or with the VS Code Ports panel.
