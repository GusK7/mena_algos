import pandas as pd
import numpy as np
import talib

# Supersmoother
# Equation 3-3
function supersmoother(x::Array{Float64}; n::Int64=10)::Array{Float64}
a = exp(-1.414*3.14159 / n)
b = 2 * a * cosd(1.414 * 180 / n)
c2 = b
c3 = -a * a
c1 = 1 - c2 - c3
@assert n<size(x,1) && n>0 "Argument n out of bounds."
Super = zeros(x)
 @inbounds for i = 3:length(x)
Super[i] = c1 * (x[i] + x[i-1]) / 2 + c2 * Super[i-1] + c3 * Super[i-2]
end
return Super
end

#Average True Range  
def ATR(df, n):  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n), name = 'ATR_' + str(n))  
    df = df.join(ATR)  
    return df


#Keltner Channel  
def KELCH(df, n):  
    KelChM = pd.Series(pd.rolling_mean((df['High'] + df['Low'] + df['Close']) / 3, n), name = 'KelChM_' + str(n))  
    KelChU = pd.Series(pd.rolling_mean((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n), name = 'KelChU_' + str(n))  
    KelChD = pd.Series(pd.rolling_mean((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n), name = 'KelChD_' + str(n))  
    df = df.join(KelChM)  
    df = df.join(KelChU)  
    df = df.join(KelChD)  
    return df

    function HpLpRoofingFilter(x::Array{Float64})::Array{Float64}
        @assert n<size(x,1) && n>0 "Argument n out of bounds."
# Highpass filter cyclic components whose periods are shorter than 48 bars
alpha1 = (cosd(360 / 48) + sind(360 / 48) - 1) / cosd(360 / 48)
HP = zeros(x)
@inbounds for i = 2:size(x,1)
    HP[i] = (1 - alpha1 / 2)*(x[i] - x[i-1]) + (1 - alpha1)*HP[i-1]
end
# Smooth with a Super Smoother Filter from equation 3-3
a1 = exp(-1.414*3.14159 / 10)  # may wish to make this an argument in function
b1 = 2*a1*cosd(1.414*180 / 10) # may wish to make this an argument in function
c2 = b1
c3 = -a1*a1
c1 = 1 - c2 - c3
LP_HP_Filt = zeros(x)
@inbounds for i = 3:size(x,1)
LP_HP_Filt[i] = c1*(HP[i] + HP[i-1]) / 2 + c2*LP_HP_Filt[i-1] + c3*LP_HP_Filt[i-2]
end
return LP_HP_Filt
end


# Ehlers Stochastic CG Oscillator [LazyBear]
# Implement trading view indicator

def ESCGO(df, p_len=8):
    d_len = len(df)

    hl2 = np.array((df.high + df.low) / 2)

    nm = [0] * d_len
    dm = [0] * d_len
    cg = [0] * d_len
    v1 = [0] * d_len
    v2 = [0] * d_len
    v3 = [0] * d_len
    t = [0] * d_len

    for i in range(p_len-1, d_len):
        for j in range(0, p_len):
            nm[i] += (j + 1) * hl2[i - j]
            dm[i] += hl2[i - j]

        cg[i] = -nm[i] / dm[i] + (p_len + 1) / 2.0 if dm[i] != 0 else 0

    cg = np.array(cg)
    min_value, max_value = talib.MINMAX(cg, timeperiod=p_len)

    for i in range(p_len-1, d_len):
        v1[i] = (cg[i] - min_value[i]) / (max_value[i] - min_value[i]) if max_value[i] is not min_value[i] else 0
        v2[i] = (4 * v1[i] + 3 * v1[i - 1] + 2 * v1[i - 2] + v1[i - 3]) / 10.0
        v3[i] = 2 * (v2[i] - 0.5)
        t[i] = (0.96 * ((v3[i - 1]) + 0.02))

    return v3, t