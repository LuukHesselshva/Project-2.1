V_in = 3.3
R1 = 30000  # Weerstand R1 in ohm
R2 = 10000  # Weerstand R2 in ohm

def spanningsdeler(V_in, R1, R2):
    V_out = V_in * (R2 / (R1 + R2))
    return V_out

print("Uitgangsspanning van de spanningsdeler is:", spanningsdeler(V_in, R1, R2), "V")