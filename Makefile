.PHONY: deploy check test clean down

# Default run_id if none specified
RUN_ID ?= latest

# Test payload for API testing
TEST_PAYLOAD := {"Customer ID": "1234-ABCD", "Age": 37, "Gender": "Female", "Married": "Yes", "Number of Dependents": 0, "Number of Referrals": 2, "Tenure in Months": 9, "Offer": "None", "Phone Service": "Yes", "Avg Monthly Long Distance Charges": 42.39, "Multiple Lines": "No", "Internet Service": "Yes", "Internet Type": "Cable", "Avg Monthly GB Download": 16.0, "Online Security": "No", "Online Backup": "Yes", "Device Protection Plan": "No", "Premium Tech Support": "Yes", "Streaming TV": "Yes", "Streaming Movies": "No", "Streaming Music": "No", "Unlimited Data": "Yes", "Contract": "One Year", "Paperless Billing": "Yes", "Payment Method": "Credit Card", "Monthly Charge": 65.6, "Total Charges": 593.3, "Total Refunds": 0.0, "Total Extra Data Charges": 0.0, "Total Long Distance Charges": 381.51, "Total Revenue": 974.81, "Latitude": 34.827662, "Longitude": -118.999073}

deploy:
	@echo "Deploying model: $(RUN_ID)"
	docker build --build-arg RUN_ID=$(RUN_ID) -f Dockerfile.api -t churn-prediction:latest .
	docker tag churn-prediction:latest localhost:5003/churn-prediction:latest
	docker push localhost:5003/churn-prediction:latest
	docker compose up -d
	sleep 5
	@echo "Checking health..."
	@curl -f http://localhost:8000/health > /dev/null 2>&1 || (echo "Service health check failed" && docker compose logs churn-prediction && exit 1)
	@echo "Deployment complete!"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"

check:
	@echo "Checking API health..."
	@curl -f http://localhost:8000/health > /dev/null 2>&1 || (echo "Health check failed" && exit 1)
	@echo "API is healthy"

test:
	@echo "Testing API with real data..."
	@echo "Request payload:"
	@echo '$(TEST_PAYLOAD)' | jq '.'
	@echo ""
	@echo "Sending request to API..."
	@echo "Response:"
	@curl -s -X POST "http://localhost:8000/predict" \
		-H "Content-Type: application/json" \
		-d '$(TEST_PAYLOAD)' \
		| jq '.'
	@echo ""
	@echo "Test completed!"

clean:
	docker compose down
	docker rmi churn-prediction:latest 2>/dev/null || true

down:
	docker compose down
