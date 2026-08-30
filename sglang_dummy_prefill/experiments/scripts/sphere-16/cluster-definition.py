from mergexp import *

net = Network('A', addressing == ipv4)

sgpu0 = net.node(
    'sgpu0',
    metal == True,
    image == "cuda126-ubuntu2404",
    host == "sgpu0"
)

sgpu2 = net.node(
    'sgpu2',
    metal == True,
    image == "cuda126-ubuntu2404",
    host == "sgpu2"
)

sgpu3 = net.node(
    'sgpu3',
    metal == True,
    image == "cuda126-ubuntu2404",
    host == "sgpu3"
)

sgpu4 = net.node(
    'sgpu4',
    metal == True,
    image == "cuda126-ubuntu2404",
    host == "sgpu4"
)


sgpu6 = net.node(
    'sgpu6',
    metal == True,
    image == "cuda126-ubuntu2404",
    host == "sgpu6"
)

sgpu7 = net.node(
    'sgpu7',
    metal == True,
    image == "cuda126-ubuntu2404",
    host == "sgpu7"
)

sgpu8 = net.node(
    'sgpu8',
    metal == True,
    image == "cuda126-ubuntu2404",
    host == "sgpu8"
)

sgpu9 = net.node(
    'sgpu9',
    metal == True,
    image == "cuda126-ubuntu2404",
    host == "sgpu9"
)

nodes = [sgpu0, sgpu2, sgpu3, sgpu4, sgpu6, sgpu7, sgpu8, sgpu9]

# Single LAN connecting everyone
lan = net.connect(nodes)

# Assign IPs in the same subnet
for idx, node in enumerate(nodes):
    lan[node].socket.addrs = ip4(f"10.0.0.{idx + 1}/24")

experiment(net)