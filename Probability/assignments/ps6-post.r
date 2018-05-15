# Problem 2 ----
lw = 2;
a = c(1, 10, 50, 500, 30)
b = c(1,10, 50, 500, 70)
maxind = 4 #index of maximum indices
col = c(rgb(0,1,0), rgb(0,1,1), rgb(1,1,0), rgb(1,0,0), rgb(0,0,1))
nprior = 5

print('2b') 
#----
par(mar=c(4,2,1,1))

th = seq(0,1,.0005)
n = 0;
m = 0;
xlab = expression(theta)
j = maxind
plot(th,dbeta(th,a[j]+n,b[j]+m), type='l',col=col[j], lwd=lw, 
     xlab=xlab,ylab='',cex.lab=2)
for (j in 1:(nprior-1))
    lines(th,dbeta(th,a[j]+n,b[j]+m), col=col[j], lwd=lw)

abline(v=.5,col='red',lty='dashed')
#Add a legend
legend(.55,25, 
       c('Beta(1,1)', 'Beta(10,10)', 'Beta(50,50', 'Beta(500,500)', 'Beta(30,70)'),
       lty=c(1,1), lwd=2, col=col)

print('2d') 
#----
print('2d')
if (doprob2 == 2)
th = seq(0,1,.001)
n = 140;
m = 110;
j = maxind
r = a[j]+n;
s = b[j]+m;    
xlab = expression(theta)
plot(th,dbeta(th,r,s), type='n', col=col[j],lwd=lw,xlab=xlab,ylab='',
     cex.lab = 2)
for (j in 1:5)
{
  r = a[j]+n;
  s = b[j]+m;
  lines(th,dbeta(th,r,s), col=col[j],lwd=lw)
  v = dbeta(.6,r,s);
  st = sprintf('Beta(%d,%d) at .6 = %f', r, s, v);
  print(st)
}
abline(v=140/250, col='red', lty='dashed')
abline(v=.5, col='red', lty='dashed')
legend(.6,25, 
       c('Beta(1,1)', 'Beta(10,10)', 'Beta(50,50', 'Beta(500,500)', 'Beta(30,70)'),
       lty=c(1,1), lwd=2, col=col)

