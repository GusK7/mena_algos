import pandas as pd

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