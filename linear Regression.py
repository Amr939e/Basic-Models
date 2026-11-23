import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



class DataFrame:
    def __init__(self,file_path,sheet_name,skiprows,header, usecols, nrows,
                 window, column_names = None, Time = False):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.usecols = usecols
        self.skiprows = skiprows
        self.nrows = nrows
        self.header = header
        self.column_names = column_names
        self.Time = Time
        self.window = window


    def DF_Regression(self):
        df = pd.read_excel(self.file_path,
                    sheet_name = self.sheet_name,
                    skiprows = self.skiprows, nrows = self.nrows,
                    header = self.header, usecols = self.usecols)
        if self.Time:
            if self.column_names:
                df.columns = self.column_names
                print(f"Renamed columns to: {df.columns.tolist()}")
                col_d = df.columns[0]
                df = df.sort_values(col_d)
                mat_x = df.columns[1]
                col_y = df.columns[-1]
            else:
                col_d = df.columns[0]
                df = df.sort_values(col_d)
                mat_x = df.colums[1]
                col_y = df.columns[-1]
        else:
            mat_x = df.columns[0]
            col_y = df.columns[-1]
      
        model = LinearRegression()
        model.fit(df[[mat_x]],df[col_y])

        intercept = model.intercept_
        slope = model.coef_[0]
        r2 = model.score(df[[mat_x]],df[col_y])
        corr = np.corrcoef(df[mat_x],df[col_y])[0,1]

        predicted_y = model.predict(df[[mat_x]])
        residuals = predicted_y - df[col_y]
        mse = mean_squared_error(df[col_y], predicted_y)
        from scipy import stats

        n = len(df)
        df_reg = n - 2  
        t = slope / (np.sqrt(mse / np.var(df[mat_x], ddof=1)) / np.sqrt(n - 1))
        p_value = 2 * (1 - stats.t.cdf(abs(t), df_reg))

        print(f"intercept: {intercept:.4f}, slope: {slope:.4f}, R^2: {r2:.4f}, Correlation: {corr:.4f}, "
              f"Mean squared error: {mse:.4f}, "
              f"t-statistic: {t:.4f},p-value: {p_value:.4f}")

        fig1, axes1 = plt.subplots(1, 3, figsize=(20, 5))
        plt.subplots_adjust(hspace=0.4, wspace = 0.3)

        # Scatter plot with regression 
        ax1 = axes1[0]
        ax1.scatter(df[mat_x], df[col_y],40, 'b')
        ax1.plot(df[mat_x], predicted_y, 'g--',linewidth=1, label='Regression')
        ax1.set_xlabel(f'{mat_x} (%)')
        ax1.set_ylabel(f'{col_y} (%)')
        ax1.set_title(f'{mat_x} vs {col_y} with Regression Line')
        ax1.legend()
        ax1.grid(True, alpha=0.3)


        # Residuals
        ax2 = axes1[1]
        ax2.scatter(predicted_y, residuals,40, 'b', alpha=0.6)
        ax2.axhline(y=0)
        ax2.set_xlabel(f'Predicted {col_y} (%)')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)

        # Actual vs Predicted
        ax3 = axes1[2]
        ax3.scatter(df[col_y], predicted_y, 40, 'y', alpha=0.6)
        ax3.plot([df[col_y].min(), df[col_y].max()],
                       [predicted_y.min(), predicted_y.max()],
                 'r--', linewidth=1, label='Perfect Prediction')
        ax3.set_xlabel(f' {col_y} (%) ')
        ax3.set_ylabel(f'Predicted {col_y} (%)')
        ax3.set_title('Actual vs Predicted')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.show()
        if self.Time:
            fig3, axes3 = plt.subplots(1, 3, figsize=(20, 5))
            a1 = np.concatenate((df[[mat_x]], df[[col_y]]), axis=1)
            a2 = []
            v = 5
            for i in range(v, n):
                m = a1[0:i, :]
                c = np.corrcoef(m[:, 0], m[:, 1])[0, 1]
                a2.append(c)
            ax4 = axes3[0]
            ax4.plot(df[col_d][v:], a2, 'r-')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Expanding Correlation')
            ax4.set_title('Expanding Correlation over Time')
            ax4.grid(True, alpha=0.3)

            rolling_corr = df[col_y].rolling(self.window).corr(df[mat_x])
            ax7 = axes3[1]
            ax7.plot(df[col_d], rolling_corr, 'r-')
            ax7.set_xlabel('Time')
            ax7.set_ylabel('Rolling Correlation')
            ax7.set_title('Rolling Correlation over Time')
            ax7.grid(True, alpha=0.3)

            rolling_mse = []
            for i in range(self.window, self.nrows):
                mse1 = mean_squared_error(df[col_y][i - self.window:i], predicted_y[i - self.window:i])
                rolling_mse.append(mse1)
            ax8 = axes3[2]
            ax8.plot(df[col_d][self.window:], rolling_mse, 'r-')
            ax8.set_xlabel('Time')
            ax8.set_ylabel('Rolling Mean Squared Error')
            ax8.set_title('Rolling Mean Squared Error over Time')
            ax8.grid(True, alpha=0.3)

            import matplotlib.dates as mdates
            for ax in [ax4,ax7, ax8]:  # Only axes that have dates
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax.tick_params(axis='x', labelsize=8, rotation=45)
            plt.show()

            fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
            plt.subplots_adjust(hspace=0.4, wspace=0.3)


            rolling_corr = df[col_y].rolling(self.window).corr(df[mat_x])
            ax7 = axes2[0]
            ax7.plot(df[col_d], rolling_corr, 'r-')
            ax7.set_xlabel('Time')
            ax7.set_ylabel('Rolling Correlation')
            ax7.set_title('Rolling Correlation over Time')
            ax7.grid(True, alpha=0.3)

            rolling_mse = []
            for i in range(self.window, self.nrows):
                mse1 = mean_squared_error(df[col_y][i - self.window:i], predicted_y[i - self.window:i])
                rolling_mse.append(mse1)
            ax8 = axes2[1]
            ax8.plot(df[col_d][self.window:], rolling_mse, 'r-')
            ax8.set_xlabel('Time')
            ax8.set_ylabel('Rolling Mean Squared Error')
            ax8.set_title('Rolling Mean Squared Error over Time')
            ax8.grid(True, alpha=0.3)


            for ax in [ ax7, ax8]:  # Only axes that have dates
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax.tick_params(axis='x', labelsize=8, rotation=45)

            plt.show()

#UK_CN_GB = DataFrame(r'C:\Users\night\OneDrive\Documents\Chinese\Projects\Data.xlsx',
                     #'UK.China2Y',
                      #  12, 0, [0,7,1],
                       # 37,8,['Date1','PriceCN','PriceUK'], Time = True)
#DataFrame.DF_Regression(UK_CN_GB)

#UK_AM_GB = DataFrame(r'C:\Users\night\OneDrive\Documents\Chinese\Projects\Data.xlsx',
                     #'UK.China2Y',
                      #  8, 0, [0,1,13],
                       # 41,12,['Date1','PriceAM','PriceUK'] ,Time = True)
#DataFrame.DF_Regression(UK_AM_GB)
