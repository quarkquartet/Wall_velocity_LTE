#!/usr/bin/env python
# coding: utf-8

# # Bubble wall velocity under LTE

# In[1]:


import numpy as np
from numpy import linalg as la
from scipy import integrate
from scipy import optimize
from scipy import interpolate
import scipy as sp
from matplotlib import pyplot as plt
import plotter as pl
from helperFunctions import derivative, alpha_p, cs_sq, dYdtau, dvTdxi, r_func, μ, w

Abs = np.abs
Log = np.log
Log10 = np.log10
Pi = np.pi
Sqrt = np.sqrt
Exp = np.exp
Cos = np.cos
Sin = np.sin
Sech = lambda x: 1/np.cosh(x)
Tanh = np.tanh
ArcSin = np.arcsin
ArcTanh = np.arctanh
Arg = np.angle
BesselK = sp.special.kv
Zeta = sp.special.zeta
HeavisideTheta = lambda x: np.heaviside(x, 0)


def Plot(fun, xminmax, n=100,xfun=np.linspace, xlog=False, ylog=False):
    xlist = xfun(xminmax[0], xminmax[1], n)
    ylist = [fun(x) for x in xlist]
    plt.plot(xlist, ylist)
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')

# ## Prepare

# Import model

# In[2]:


import SM_model as m

# In[3]:


mod = m.SM(1,0.007285228,636.8644639563023)

# In[4]:


mod.findTc()
mod.findTn()

# In[5]:


mod.findα()
mod.findβH()

# In[6]:


Vtot=mod.Vtot
hv = np.array([0.0])
lv = np.array([mod.Tnvev])
Tnuc = mod.Tn

# In[7]:


from helperFunctions import a

# ## Solving the boundary conditions for deflagration

# In[12]:


def p(V, T, vev):
    v = optimize.fmin(V, vev, args=(T,), disp=0)
    return -V(v, T)

def e(V, T, vev):
    v = optimize.fmin(V, vev, args=(T,), disp=0)
    def VT(T):
        return V(v, T)
    return - T * derivative(VT, T) + VT(T)

def ω(V, T, vev):
    v = optimize.fmin(V, vev, args=(T,), disp=0)
    def VT(T):
        return V(v, T)
    return - T * derivative(VT, T) 

# In[77]:


def match(vp,vm,Tp,Tm, high_vev, low_vev):
    r = r_func(Vtot, Tp, Tm, high_vev, low_vev)
    αp = alpha_p(Vtot, Tp, Tm, high_vev, low_vev)
    vpvm = 1-(1-3*αp)*r
    vpvm = vpvm/(3-3*(1+αp)*r)
    ratio = 3 + (1-3*αp)*r
    ratio = ratio/(1+3*(1+αp)*r)
    return [vp*vm - vpvm, vp/vm - ratio]

def match_pe(vp,vm, Tp, Tm, hv, lv):
    pp = p(Vtot, Tp, hv)
    pm = p(Vtot, Tm, lv)
    ep = e(Vtot, Tp, hv)
    em = e(Vtot, Tm, lv)
    vpvm = (pp - pm)/(ep - em)
    ratio = (em + pp)/(ep + pm)
    return np.array([vp* vm- vpvm, vp/vm - ratio])

def match_T(vp,vm, Tp, Tm, hv, lv):
    pp = p(Vtot, Tp, hv)
    pm = p(Vtot, Tm, lv)
    ωp = ω(Vtot, Tp, hv)
    ωm = ω(Vtot, Tm, lv)
    γp = 1/Sqrt(1-vp**2)
    γm = 1/Sqrt(1-vm**2)
    T33p = ωp * vp**2 * γp**2 + pp
    T33m = ωm * vm**2 * γm**2 + pm
    T30p = ωp * vp * γp**2
    T30m = ωm * vm * γm**2
    return np.array([T33p - T33m, T30p - T30m])

def find_Tsh(Tm, vw, type='def'):
    if type=='def':
        guess_sol = optimize.fsolve(lambda x:match(x[0], vw, x[1], Tm,hv, lv),[vw*0.8,Tnuc])
    elif type=='hyb':
        guess_sol = optimize.fsolve(lambda x:match(x[0], cs_sq(Vtot, Tm, lv)**0.5, x[1], Tm,hv, lv),[0.5,Tnuc])

    # Integrate outside the wall to the shock-wave front
    try:
        vsol=integrate.solve_ivp(dYdtau, (10,0.01), np.array([μ(vw, guess_sol[0]), guess_sol[1], vw]),t_eval=np.linspace(10,0.01,1000),method='DOP853',args=(Vtot, hv))
        xi_max = vsol.y[2].max()
        xi_max_index = vsol.y[2].argmax()
        v_prof = interpolate.interp1d(vsol.y[2][0:xi_max_index+1],vsol.y[0][0:xi_max_index+1])
        T_prof = interpolate.interp1d(vsol.y[2][0:xi_max_index+1],vsol.y[1][0:xi_max_index+1])
        try:
            xsh=optimize.brentq(lambda x: μ(x, v_prof(x))*x - cs_sq(Vtot, T_prof(x), hv), vw, xi_max)
        except:
            xsh = xi_max
    except:
        vTsol = integrate.solve_ivp(dvTdxi, (vw, 1), np.array([μ(vw, guess_sol[0]), guess_sol[1]]), t_eval=np.linspace(vw, 1, 500), method='DOP853', args=(Vtot, hv))
        v_prof = interpolate.interp1d(vTsol.t, vTsol.y[0], kind='cubic')
        T_prof = interpolate.interp1d(vTsol.t, vTsol.y[1], kind='cubic')
        xsh = optimize.brentq(lambda x: μ(x, v_prof(x))*x - cs_sq(Vtot, T_prof(x), hv), vw, 1)

    def sh_match_2(Tshp):
        ap = alpha_p(Vtot, Tshp, T_prof(xsh), hv, hv)
        r = r_func(Vtot, Tshp, T_prof(xsh), hv, hv)
        vp = xsh
        vm = μ(xsh, v_prof(xsh))
        ratio = 3 + (1-3*ap)*r
        ratio = ratio/(1+3*(1+ap)*r)
        return vp/vm - ratio
    Tshp = optimize.newton(sh_match_2, T_prof(xsh))
    return Tshp

# In[9]:


Tnuc

# In[41]:


find_Tsh(53, 0.1)

# In[47]:


Tm = 53
vw = 0.55
print(optimize.fsolve(lambda x:match(x[0], vw, x[1], Tm,hv, lv),[0.4,Tnuc]))
print(optimize.fsolve(lambda x:match_pe(x[0], vw, x[1], Tm,hv, lv),[0.4,Tnuc]))
print(optimize.fsolve(lambda x:match_T(x[0], vw, x[1], Tm,hv, lv),[0.4,Tnuc]))

# In[48]:


guess_sol = optimize.fsolve(lambda x:match(x[0], vw, x[1], Tm,hv, lv),[0.4,Tnuc])

# In[49]:


vsol=integrate.solve_ivp(dYdtau, (10,0.01), np.array([μ(vw, guess_sol[0]), guess_sol[1], vw]),t_eval=np.linspace(10,0.01,1000),method='DOP853',args=(Vtot, hv))
xi_max = vsol.y[2].max()
xi_max_index = vsol.y[2].argmax()
v_prof = interpolate.interp1d(vsol.y[2][0:xi_max_index+1],vsol.y[0][0:xi_max_index+1])
T_prof = interpolate.interp1d(vsol.y[2][0:xi_max_index+1],vsol.y[1][0:xi_max_index+1])
try:
    xsh=optimize.brentq(lambda x: μ(x, v_prof(x))*x - cs_sq(Vtot, T_prof(x), hv), vw, xi_max)
except:
    xsh = xi_max

# In[17]:


xsh

# In[50]:


def sh_match(Tshp):
    ap = alpha_p(Vtot, Tshp, T_prof(xsh), hv, hv)
    r = r_func(Vtot, Tshp, T_prof(xsh), hv, hv)
    vp = xsh
    vm = μ(xsh, v_prof(xsh))
    vpvm = 1-(1-3*ap)*r
    vpvm = vpvm/(3-3*(1+ap)*r)
    return vpvm - vp*vm

def sh_match_2(Tshp):
    ap = alpha_p(Vtot, Tshp, T_prof(xsh), hv, hv)
    r = r_func(Vtot, Tshp, T_prof(xsh), hv, hv)
    vp = xsh
    vm = μ(xsh, v_prof(xsh))
    ratio = 3 + (1-3*ap)*r
    ratio = ratio/(1+3*(1+ap)*r)
    return vp/vm - ratio

# 

# In[27]:


μ(xsh, v_prof(xsh))

# In[74]:


r_func(Vtot, 52.82540856769438, 54.37164544, hv, hv)

# In[23]:


sh_match(53)

# In[29]:


Plot(sh_match, (52,55), n=30)

# In[38]:


optimize.newton(sh_match_2, 53.6)

# In[33]:


T_prof(xsh)

# In[54]:


find_Tsh(53.3,0.1)

# ## Solve hybrid boundary conditions

# In[302]:


vw = 0.65

# In[79]:


Tm=53
cs_sq(Vtot, Tm, lv)**0.5

# In[82]:


guess_sol = optimize.fsolve(lambda x:match(x[0], cs_sq(Vtot, Tm, lv)**0.5, x[1], Tm,hv, lv),[0.5,Tnuc])

# In[83]:


guess_sol

# In[84]:


ini = np.array([μ(vw, guess_sol[0]), guess_sol[1], vw])

# In[91]:


sol=integrate.solve_ivp(dYdtau, (10,1), ini, args=(Vtot, hv), t_eval=np.linspace(10, 1,100))

# In[92]:


sol

# In[93]:


plt.plot(sol.y[2], sol.y[0])

# In[94]:


plt.plot(sol.y[2], sol.y[1])

# In[95]:


xi_max = sol.y[2].max()
xi_max_index = sol.y[2].argmax()

# In[96]:


v_prof = interpolate.interp1d(sol.y[2][0:xi_max_index+1],sol.y[0][0:xi_max_index+1])
T_prof = interpolate.interp1d(sol.y[2][0:xi_max_index+1],sol.y[1][0:xi_max_index+1])

# In[97]:


Plot(v_prof, (0.6, xi_max))

# In[98]:


Plot(T_prof, (0.6, xi_max))

# In[106]:


xsh = optimize.brentq(lambda x: μ(x, v_prof(x)) * x - cs_sq(Vtot, T_prof(x), hv), vw, xi_max)
print(xsh)

# In[109]:


μ(xsh, v_prof(xsh))

# In[111]:


def sh_match_2(Tshp):
    ap = alpha_p(Vtot, Tshp, T_prof(xsh), hv, hv)
    r = r_func(Vtot, Tshp, T_prof(xsh), hv, hv)
    vp = xsh
    vm = μ(xsh, v_prof(xsh))
    ratio = 3 + (1-3*ap)*r
    ratio = ratio/(1+3*(1+ap)*r)
    return vp/vm - ratio

# In[112]:


optimize.newton(sh_match_2, T_prof(xsh))

# In[110]:


find_Tsh(53, 0.6, 'hyb')

# In[118]:


find_Tsh(55.925, 0.6, 'hyb')

# In[128]:


Ttest=55.925
optimize.fsolve(lambda x:match(x[0], cs_sq(Vtot, Ttest, lv)**0.5, x[1], Ttest,hv, lv),[0.1,Tnuc+3])

# In[304]:


Tm = optimize.newton(lambda T: find_Tsh(T, 0.65, 'hyb') - Tnuc, 56)
print(Tm)

# In[306]:


vp, Tp = optimize.fsolve(lambda x:match(x[0], cs_sq(Vtot, Tm, lv)**0.5, x[1], Tm,hv, lv),[0.5,Tm + 2])
print(vp)
print(Tp)

# In[307]:


vm =cs_sq(Vtot, Tm, lv)**0.5
print(vm)
print(Tm)
vp, Tp = optimize.fsolve(lambda x:match_T(x[0], vm, x[1], Tm,hv, lv),[0.5,Tm+2])
print(vp)
print(Tp)

# ## Solve detonation boundary conditions

# In[312]:


from helperFunctions import vJ

# In[316]:


vw=0.7
gsol=optimize.fsolve(lambda x:match(vw, x[0], Tnuc, x[1], hv, lv),[vw*0.7,Tnuc+2])

# In[317]:


gsol

# In[318]:


vJ(alpha_p(Vtot, Tnuc, gsol[1], hv, lv))

# In[335]:


vw=0.64
gsol=optimize.fsolve(lambda x:match(vw, x[0], Tnuc, x[1], hv, lv),[vw*0.9,Tnuc+2])
print(gsol)
print(vJ(alpha_p(Vtot, Tnuc, gsol[1], hv, lv)))

# In[344]:


def test_vJ(vw):
    gsol=optimize.fsolve(lambda x:match(vw, x[0], Tnuc, x[1], hv, lv),[vw*0.9,Tnuc+2])
    return vJ(alpha_p(Vtot, Tnuc, gsol[1], hv, lv))

# In[346]:


Plot(test_vJ, (0.64,0.75), n=20)
plt.plot(np.linspace(0.64,0.75,20), np.linspace(0.64,0.75,20))

# In[348]:


vwmax = 1.0
eps = 0.01
for i in range(1000):
    vw = vwmax - i * eps
    if test_vJ(vw) > vw:
        vwmin = vw
        break
vJvalue = optimize.brentq(lambda vw:test_vJ(vw) - vw, vwmin, vwmin+eps)

# In[349]:


vJvalue

# In[325]:


vw=0.7
vm,Tm =optimize.fsolve(lambda x:match(vw, x[0], Tnuc, x[1], hv, lv),[vw*0.7,Tnuc+2])
vp = vw
Tp = Tnuc
lv_new = optimize.fmin(Vtot, lv, args=(Tm,),disp=0)

# In[323]:


- Tp/Sqrt(1-vp**2) + Tm/Sqrt(1-vm**2) # entropy difference, should be negative

# In[331]:


Lh = 0.1
npoints = 100
z_range = np.linspace(-8*Lh, 5*Lh, npoints)
T_sol = np.zeros((npoints,))
for i in range(npoints):
    T33min = optimize.minimize(lambda T: T33(T[0], z_range[i], Lh), Tp, method='Nelder-Mead', bounds = [(40, 90)])
    if T33min.fun > 0:
        T_sol[i]=T33min.x[0]
    else:
        try:
            s = optimize.newton(lambda T: T33(T, z_range[i], Lh), Tp)
        except:
            s = optimize.fsolve(lambda T: T33(T[0], z_range[i], Lh), Tp)[0]
        T_sol[i] = s

Tsol_list.append(T_sol)
hvalues = h_profile(z_range, Lh)
hprime = np.vectorize(lambda z: -0.5*(h0*Sech(z/Lh)**2)/Lh)
d2zh = np.vectorize(lambda z: (h0*Sech(z/Lh)**2*Tanh(z/Lh))/Lh**2)
Eh = np.array([mod.gradV([hvalues[i]], T_sol[i]) - d2zh(z_range[i])  for i in range(npoints)]).reshape((-1,))
    
inte = - Eh * hprime(z_range)

# In[332]:


plt.plot(z_range, T_sol)

# In[333]:


plt.plot(z_range, Eh)

# In[334]:


plt.plot(z_range, inte)

# In[350]:


vwdet = np.linspace(vJvalue*1.01, 0.99, 5).tolist()
Pdet = []
sdiffdet = []

# In[351]:


for vw in vwdet:
    print("vw = " + str(vw))
    vm,Tm =optimize.fsolve(lambda x:match(vw, x[0], Tnuc, x[1], hv, lv),[vw*0.7,Tnuc+2])
    vp = vw
    Tp = Tnuc
    lv_new = optimize.fmin(Vtot, lv, args=(Tm,),disp=0)
    h0 = lv_new

    sdiffdet.append(- Tp/Sqrt(1-vp**2) + Tm/Sqrt(1-vm**2))

    def h_profile(z, Lh):
        z = np.asanyarray(z)
        hz = 0.5*h0*(1-np.tanh(z/Lh))
        return hz
    c1 = w(Vtot, Tm, lv_new) * vm/(1-vm**2)
    s1=c1
    c2=-Vtot(lv_new, Tm)+ w(Vtot, Tm, lv_new) * vm**2 /(1-vm**2)
    s2=c2
    def T33(T,z, Lh):
        derh = derivative(lambda zvalue: h_profile(zvalue,Lh),z)
        field_value = [h_profile(z, Lh)]
        return (0.5*derh**2 - Vtot(field_value, T) - 0.5*w(Vtot, T, field_value) + 0.5*(4*s1**2 + w(Vtot, T, field_value)**2)**0.5 - s2)/1e6
    
    def moments(Lh):
        npoints = 100
        z_range = np.linspace(-8*Lh, 5*Lh, npoints)
        T_sol = np.zeros((npoints,))
        for i in range(npoints):
            T33min = optimize.minimize(lambda T: T33(T[0], z_range[i], Lh), Tp, method='Nelder-Mead', bounds = [(40, 90)])
            if T33min.fun > 0:
                T_sol[i]=T33min.x[0]
            else:
                try:
                    s = optimize.newton(lambda T: T33(T, z_range[i], Lh), Tp)
                except:
                    s = optimize.fsolve(lambda T: T33(T[0], z_range[i], Lh), Tp)[0]
                T_sol[i] = s

        hvalues = h_profile(z_range, Lh)
        hprime = np.vectorize(lambda z: -0.5*(h0*Sech(z/Lh)**2)/Lh)
        d2zh = np.vectorize(lambda z: (h0*Sech(z/Lh)**2*Tanh(z/Lh))/Lh**2)
        Eh = np.array([mod.gradV([hvalues[i]], T_sol[i]) - d2zh(z_range[i])  for i in range(npoints)]).reshape((-1,))

        Ph = np.trapz(- Eh * hprime(z_range), z_range)
        Gh = np.trapz( Eh * hprime(z_range) *(2*h_profile(z_range, Lh)/h0 - 1) , z_range)
        return np.array([Ph, Gh])/1e6
    
    print('Solving moments.')
    Lsol = optimize.newton(lambda L: moments(L)[-1], 0.2)
    print('Moment solved, Lh = ' + str(Lsol))
    P = moments(Lsol)[0]
    Pdet.append(P)


# In[353]:


Pdet

# In[354]:


sdiffdet

# In[355]:


vwlist

# In[356]:


Plist.tolist()

# In[358]:


Plist

# In[357]:


sdifflist

# In[359]:


Pfinal = Plist.tolist() + Pdet
sdifffinal = sdifflist.tolist() + sdiffdet

# In[361]:


vwfinal = vwlist + vwdet.tolist()

# In[363]:


plt.plot(vwfinal, Pfinal)
plt.xlabel(r'$v_w$')
plt.ylabel(r'$P_{\rm tot}$')

# In[365]:


plt.plot(vwfinal[0:-1], sdifffinal[0:-1])
plt.xlabel(r'$v_w$')
plt.ylabel(r'$\Delta (s \gamma v)$')

# ## Solve the temperature profile and the moments

# In[326]:


h0 = lv_new

def h_profile(z, Lh):
    z = np.asanyarray(z)
    hz = 0.5*h0*(1-np.tanh(z/Lh))
    return hz
c1 = w(Vtot, Tm, lv_new) * vm/(1-vm**2)
s1=c1
c2=-Vtot(lv_new, Tm)+ w(Vtot, Tm, lv_new) * vm**2 /(1-vm**2)
s2=c2

# In[327]:


def T33(T,z, Lh):
    derh = derivative(lambda zvalue: h_profile(zvalue,Lh),z)
    field_value = [h_profile(z, Lh)]
    return (0.5*derh**2 - Vtot(field_value, T) - 0.5*w(Vtot, T, field_value) + 0.5*(4*s1**2 + w(Vtot, T, field_value)**2)**0.5 - s2)/1e6

# In[286]:


vwlist = [0.4,0.5,0.6]
Tsol_list = []
Ehlist = []
intelist = []
for vw in vwlist:
    # Test solution type. Start from deflagration to test.
    Tm = optimize.newton(lambda T: find_Tsh(T, vw)-Tnuc, Tnuc)
    lv_new = mod.findMinimum(lv, Tm)
    if cs_sq(Vtot, Tm, lv_new)**0.5 > vw:
        print("Deflagration.")
        vp, Tp = optimize.fsolve(lambda x:match(x[0],vw,x[1], Tm, hv, lv),[0.3, Tnuc], xtol=1e-10)
        vm = vw
    else:
        print("Hybrid.")
        Tm = optimize.newton(lambda T: find_Tsh(T, vw, 'hyb')-Tnuc, Tnuc+1)
        lv_new = mod.findMinimum(lv, Tm)
        vm = cs_sq(Vtot, Tm,lv_new)**0.5
        vp, Tp = optimize.fsolve(lambda x:match(x[0],vm,x[1], Tm, hv, lv),[0.5, Tnuc+3], xtol=1e-10)
        print("Boundary condition found.")

    h0 = lv_new

    def h_profile(z, Lh):
        z = np.asanyarray(z)
        hz = 0.5*h0*(1-np.tanh(z/Lh))
        return hz

    c1 = w(Vtot, Tm, lv_new) * vm/(1-vm**2)
    s1=c1
    c2=-Vtot(lv_new, Tm)+ w(Vtot, Tm, lv_new) * vm**2 /(1-vm**2)
    s2=c2

    def T33(T,z, Lh):
        derh = derivative(lambda zvalue: h_profile(zvalue,Lh),z)
        field_value = [h_profile(z, Lh)]
        return (0.5*derh**2 - Vtot(field_value, T) - 0.5*w(Vtot, T, field_value) + 0.5*(4*s1**2 + w(Vtot, T, field_value)**2)**0.5 - s2)/1e6

    print('T33 prepared.')
    
    Lh = 0.1
    npoints = 100
    z_range = np.linspace(-8*Lh, 5*Lh, npoints)
    T_sol = np.zeros((npoints,))
    for i in range(npoints):
        T33min = optimize.minimize(lambda T: T33(T[0], z_range[i], Lh), Tp, method='Nelder-Mead', bounds = [(40, 90)])
        if T33min.fun > 0:
            T_sol[i]=T33min.x[0]
        else:
            try:
                s = optimize.newton(lambda T: T33(T, z_range[i], Lh), Tp)
            except:
                s = optimize.fsolve(lambda T: T33(T[0], z_range[i], Lh), Tp)[0]
            T_sol[i] = s

    Tsol_list.append(T_sol)
    hvalues = h_profile(z_range, Lh)
    hprime = np.vectorize(lambda z: -0.5*(h0*Sech(z/Lh)**2)/Lh)
    d2zh = np.vectorize(lambda z: (h0*Sech(z/Lh)**2*Tanh(z/Lh))/Lh**2)
    Eh = np.array([mod.gradV([hvalues[i]], T_sol[i]) - d2zh(z_range[i])  for i in range(npoints)]).reshape((-1,))
    
    inte = - Eh * hprime(z_range)

    Ehlist.append(Eh)
    intelist.append(inte)

# In[287]:


for i in range(3):
    plt.plot(z_range, Tsol_list[i], label=f'vw={vwlist[i]}')

plt.legend()

# In[288]:


for i in range(3):
    plt.plot(z_range, Ehlist[i], label=f'vw={vwlist[i]}')

plt.legend()

# In[289]:


for i in range(3):
    plt.plot(z_range, intelist[i], label=f'vw={vwlist[i]}')

plt.legend()

# In[268]:


np.trapz(intelist[2], z_range)/1e6

# In[328]:


def moments(Lh):
    npoints = 100
    z_range = np.linspace(-8*Lh, 5*Lh, npoints)
    T_sol = np.zeros((npoints,))
    for i in range(npoints):
        T33min = optimize.minimize(lambda T: T33(T[0], z_range[i], Lh), Tp, method='Nelder-Mead', bounds = [(40, 90)])
        if T33min.fun > 0:
            T_sol[i]=T33min.x[0]
        else:
            try:
                s = optimize.newton(lambda T: T33(T, z_range[i], Lh), Tp)
            except:
                s = optimize.fsolve(lambda T: T33(T[0], z_range[i], Lh), Tp)[0]
            T_sol[i] = s

    hvalues = h_profile(z_range, Lh)
    hprime = np.vectorize(lambda z: -0.5*(h0*Sech(z/Lh)**2)/Lh)
    d2zh = np.vectorize(lambda z: (h0*Sech(z/Lh)**2*Tanh(z/Lh))/Lh**2)
    Eh = np.array([mod.gradV([hvalues[i]], T_sol[i]) - d2zh(z_range[i])  for i in range(npoints)]).reshape((-1,))
    
    Ph = np.trapz(- Eh * hprime(z_range), z_range)
    Gh = np.trapz( Eh * hprime(z_range) *(2*h_profile(z_range, Lh)/h0 - 1) , z_range)
    return np.array([Ph, Gh])/1e6

# In[330]:


moments(0.1)

# In[293]:


moments(0.3)

# In[294]:


Plot(lambda Lh: moments(Lh)[-1], (0.08,0.3), n=10)

# In[165]:


vwlist = np.linspace(0.1,0.55,6)
Plist = np.zeros((6,))

# In[166]:


vwlist = [round(vw,2) for vw in vwlist]

# In[296]:


def Ptot(vw):
    # Test solution type. Start from deflagration to test.
    Tm = optimize.newton(lambda T: find_Tsh(T, vw)-Tnuc, Tnuc)
    lv_new = mod.findMinimum(lv, Tm)
    if cs_sq(Vtot, Tm, lv_new)**0.5 > vw:
        print("Deflagration.")
        vp, Tp = optimize.fsolve(lambda x:match(x[0],vw,x[1], Tm, hv, lv),[0.3, Tnuc], xtol=1e-10)
        vm = vw
    else:
        print("Hybrid.")
        Tm = optimize.newton(lambda T: find_Tsh(T, vw, 'hyb')-Tnuc, Tnuc+1)
        lv_new = mod.findMinimum(lv, Tm)
        vm = cs_sq(Vtot, Tm,lv_new)**0.5
        vp, Tp = optimize.fsolve(lambda x:match(x[0],vm,x[1], Tm, hv, lv),[0.5, Tnuc+3], xtol=1e-10)
        print("Boundary condition found.")


    hv_new = mod.findMinimum(hv, Tp)
    h0 = lv_new

    def h_profile(z, Lh):
        z = np.asanyarray(z)
        hz = 0.5*h0*(1-np.tanh(z/Lh))
        return hz
    c1 = w(Vtot, Tm, lv_new) * vm/(1-vm**2)
    s1=c1
    c2=-Vtot(lv_new, Tm)+ w(Vtot, Tm, lv_new) * vm**2 /(1-vm**2)
    s2=c2

    def T33(T,z, Lh):
        derh = derivative(lambda zvalue: h_profile(zvalue,Lh),z)
        field_value = [h_profile(z, Lh)]
        return (0.5*derh**2 - Vtot(field_value, T) - 0.5*w(Vtot, T, field_value) + 0.5*(4*s1**2 + w(Vtot, T, field_value)**2)**0.5 - s2)/1e6

    print('T33 prepared.')
    
    def moments(Lh):
        npoints = 100
        z_range = np.linspace(-8*Lh, 5*Lh, npoints)
        T_sol = np.zeros((npoints,))
        for i in range(npoints):
            T33min = optimize.minimize(lambda T: T33(T[0], z_range[i], Lh), Tp, method='Nelder-Mead', bounds = [(40, 90)])
            if T33min.fun > 0:
                T_sol[i]=T33min.x[0]
            else:
                try:
                    s = optimize.newton(lambda T: T33(T, z_range[i], Lh), Tp)
                except:
                    s = optimize.fsolve(lambda T: T33(T[0], z_range[i], Lh), Tp)[0]
                T_sol[i] = s

        hvalues = h_profile(z_range, Lh)
        hprime = np.vectorize(lambda z: -0.5*(h0*Sech(z/Lh)**2)/Lh)
        d2zh = np.vectorize(lambda z: (h0*Sech(z/Lh)**2*Tanh(z/Lh))/Lh**2)
        Eh = np.array([mod.gradV([hvalues[i]], T_sol[i]) - d2zh(z_range[i])  for i in range(npoints)]).reshape((-1,))
    
        Ph = np.trapz(- Eh * hprime(z_range), z_range)
        Gh = np.trapz( Eh * hprime(z_range) *(2*h_profile(z_range, Lh)/h0 - 1) , z_range)
        return np.array([Ph, Gh])/1e6
    print('Solving moments.')
    Lsol = optimize.newton(lambda L: moments(L)[-1], 0.2)
    print('Moment solved, Lh = ' + str(Lsol))
    P = moments(Lsol)[0]
    return P

# In[297]:


Ptot(0.6)

# In[298]:


vwlist = np.linspace(0.1, 0.55, 7)
vwlist = [round(vw,2) for vw in vwlist]
vwlist.append(0.6)
Plist=np.ones((8,))

# In[299]:


for i in range(len(vwlist)):
    vw = vwlist[i]
    Plist[i] = Ptot(vw)
    print(Plist[i])

# In[300]:


plt.plot(vwlist, Plist)
plt.xlabel(r'$v_w$')
plt.ylabel(r'$P_{\rm tot}$')

# In[301]:


Ptot(0.65)

# ## Solve entropy vanishing instead of P vanishing

# In[44]:


def entropy(V,T, vev):
    v = optimize.fmin(V, vev, args=(T,), disp=0)

    def VT(T):
        return V(v, T)
    
    return -derivative(VT, T, order=1)

# In[129]:


def sdiff(vw):
    print("vw = " + str(vw))
    # Test solution type. Start from deflagration to test.
    Tm = optimize.newton(lambda T: find_Tsh(T, vw)-Tnuc, Tnuc-1)
    lv_new = mod.findMinimum(lv, Tm)
    if cs_sq(Vtot, Tm, lv_new)**0.5 > vw:
        print("Deflagration.")
        vp, Tp = optimize.fsolve(lambda x:match(x[0],vw,x[1], Tm, hv, lv),[0.3, Tnuc], xtol=1e-10)
        vm = vw
    else:
        print("Hybrid.")
        Tm = optimize.newton(lambda T: find_Tsh(T, vw, 'hyb')-Tnuc, Tnuc)
        lv_new = mod.findMinimum(lv, Tm)
        vp, Tp = optimize.fsolve(lambda x:match(x[0],cs_sq(Vtot, Tm,lv_new)**0.5,x[1], Tm, hv, lv),[0.5, Tnuc+3], xtol=1e-10)
        vm = cs_sq(Vtot, Tm,lv_new)**0.5

    lv_new = mod.findMinimum(lv, Tm)

    diff = - Tp/Sqrt(1-vp**2) + Tm/Sqrt(1-vm**2)
    # diff = (- entropy(Vtot, Tp, hv) * vp/Sqrt(1-vp**2) + entropy(Vtot, Tm, lv_new) * vw/Sqrt(1-vw**2))/1e6
    return diff

# In[131]:


sdiff(0.6)

# In[145]:


vwlist = np.linspace(0.1, 0.55, 7)
vwlist = [round(vw,2) for vw in vwlist]
vwlist.append(0.6)
sdifflist = np.zeros((8,))

# In[147]:


for i in range(len(vwlist)):
    sdifflist[i] = sdiff(vwlist[i])
    print(sdifflist[i])

# In[148]:


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=16)
plt.rc('text.latex', preamble=r'\usepackage{wasysym}')
plt.plot(vwlist, sdifflist)
plt.xlabel(r'$v_w$')
plt.ylabel(r'$\Delta (s \gamma v)$')

# In[243]:


plt.plot(np.linspace(0.1, 0.55, 7),(Plist[0]/sdifflist[0])*sdifflist[0:-1])
plt.plot(vwlist, Plist)

# In[ ]:



