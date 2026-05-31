
import prediction_engine
import data_loader

def run_demo():
    print("="*60)
    print(" Disease Spread Prediction System - CLI Model Runner")
    print("="*60)
    
    # Pick a sample
    disease = "Dengue"
    region = "Maharashtra"
    days = 7
    
    print(f"\n[+] Running ensemble model for {disease} in {region}...")
    print(f"[+] Forecasting next {days} days...\n")
    
    try:
        result = prediction_engine.predict(disease, region, days)
        
        # Display Results
        print(f"{'Day':<6} | {'Ensemble':<10} | {'ARIMA':<10} | {'LSTM':<10} | {'SEIR':<10}")
        print("-" * 55)
        
        forecast = result['forecast']
        arima = result['f_arima']
        lstm = result['f_lstm']
        seir = result['f_seir']
        
        for i in range(days):
            l_val = f"{lstm[i]}" if lstm else "N/A"
            print(f"Day {i+1:<2} | {forecast[i]:<10} | {arima[i]:<10} | {l_val:<10} | {seir[i]:<10}")
        
        print(f"\n[!] Risk Assessment: {result['risk_info']['risk']}")
        print(f"[!] Alert Message: {result['alert_msg']}")
        print(f"[!] Reason: {result['explanation']}")
        
    except Exception as e:
        print(f"\n[!] Error running model: {str(e)}")

if __name__ == "__main__":
    run_demo()
