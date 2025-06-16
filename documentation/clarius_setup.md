# clarius data access setup

## Setup

### General Steps

- Download the Clarius mobile app on a mobile device
- Login and test the connection between the app and the scanner
- Download the cast API release from: https://github.com/clariusdev/cast/releases
- Make sure the `mobile device`, `scanner` and the `computer` are all connecting to the same network (or network with different names but can ping each other)

### MacOS-Python

- Copy the `libcast.dylib` binary from the cast API release to `src/data_access/clarius`
- Copy the `pyclariuscast.so` from the python version you are using (e.g. python312), from the cast API release to `src/data_access/clarius`
- Change the ip (default 192.168.1.1) and port (default 5828) in the run scripts to the actual values shown in the clarius app

## Troubleshooting

### ipv6 on MacOS

If the ip address appeared to be an ipv6 address, and connection issue happend, try following steps for troubleshooting:

- Open a terminal and run `ping6 (the scanner ipv6 address)`
- If got error relating to `UDP host`, go to `network preference > wifi > advanced > tcp/ip ` and set `configure ipv6` to `automatically`

