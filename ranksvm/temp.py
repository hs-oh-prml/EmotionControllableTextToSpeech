mu_a = 0.6
mu_b = 0.4

R = [3, 4, 1]
def qn(thetaA, thetaB, r):
    A = pow(thetaA, float(r)) * pow(1-thetaA, float(5 - r))
    B = pow(thetaB, float(r)) * pow(1-thetaB, float(5 - r))
    out =  A / (A + B) 
    return out 

def new_theta(old_thetaA, old_thetaB):
    up = 0
    down = 0
    for i in R:
        down = down + (5.0 * qn(old_thetaA, old_thetaB, i)) 
        up = up + (i * qn(old_thetaA, old_thetaB, i)) 
    outA = up / down

    up = 0
    down = 0
    for i in R:
        down = down + (5.0 * qn(old_thetaB, old_thetaA, i)) 
        up = up + (i * qn(old_thetaB, old_thetaA, i)) 
    outB = up / down

    return outA, outB

new_mu_a = 0
count = 0

while True:
    new_mu_a, new_mu_b  = new_theta(mu_a, mu_b)
    print(new_mu_a, mu_a, new_mu_b, mu_b, count)

    if abs(new_mu_a - mu_a) < 1e-10 and abs(new_mu_b - mu_b) < 1e-10: 
        print(abs(new_mu_a - mu_a))
        print(abs(new_mu_b - mu_b))
        break
    mu_a = new_mu_a
    mu_b = new_mu_b

    count = count + 1
print(new_mu_a, mu_a, new_mu_b, mu_b, count)

# a = 3
# b = 2
# c = pow(0.68, a) * pow(0.32, b) / (pow(0.68, a) * pow(0.32, b) + pow(0.37, a) * pow(0.63, b))
# d = pow(0.37, a) * pow(0.63, b) / (pow(0.68, a) * pow(0.32, b) + pow(0.37, a) * pow(0.63, b))
# print(c, d)
