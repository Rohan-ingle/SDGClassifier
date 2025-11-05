#!/bin/bash
# Quick script to view logs and metrics

set -e

LOGS_DIR="logs"
METRICS_DIR="metrics"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

show_menu() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   SDG Classifier - Logs & Metrics     ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "1) View latest training logs"
    echo "2) View latest preprocessing logs"
    echo "3) View latest evaluation logs"
    echo "4) View all error logs"
    echo "5) Show training metrics"
    echo "6) Show evaluation metrics"
    echo "7) List all log files"
    echo "8) List all metric files"
    echo "9) Tail training logs (real-time)"
    echo "10) Search logs for text"
    echo "11) Collect artifacts"
    echo "12) Clean old logs (keep last 5 days)"
    echo "0) Exit"
    echo ""
    echo -n "Choose an option: "
}

view_latest_training_logs() {
    echo -e "${BLUE}=== Latest Training Logs ===${NC}"
    LATEST=$(ls -t ${LOGS_DIR}/train_*.log 2>/dev/null | head -1)
    if [ -f "$LATEST" ]; then
        echo -e "${GREEN}File: $LATEST${NC}"
        echo ""
        cat "$LATEST"
    else
        echo -e "${RED}No training logs found${NC}"
    fi
}

view_latest_preprocess_logs() {
    echo -e "${BLUE}=== Latest Preprocessing Logs ===${NC}"
    LATEST=$(ls -t ${LOGS_DIR}/preprocess_*.log 2>/dev/null | head -1)
    if [ -f "$LATEST" ]; then
        echo -e "${GREEN}File: $LATEST${NC}"
        echo ""
        cat "$LATEST"
    else
        echo -e "${RED}No preprocessing logs found${NC}"
    fi
}

view_latest_evaluate_logs() {
    echo -e "${BLUE}=== Latest Evaluation Logs ===${NC}"
    LATEST=$(ls -t ${LOGS_DIR}/evaluate_*.log 2>/dev/null | head -1)
    if [ -f "$LATEST" ]; then
        echo -e "${GREEN}File: $LATEST${NC}"
        echo ""
        cat "$LATEST"
    else
        echo -e "${RED}No evaluation logs found${NC}"
    fi
}

view_error_logs() {
    echo -e "${RED}=== All Error Logs ===${NC}"
    if ls ${LOGS_DIR}/*_errors_*.log 1> /dev/null 2>&1; then
        for file in ${LOGS_DIR}/*_errors_*.log; do
            if [ -s "$file" ]; then
                echo -e "${YELLOW}--- $file ---${NC}"
                cat "$file"
                echo ""
            fi
        done
    else
        echo -e "${GREEN}No errors found! 🎉${NC}"
    fi
}

show_training_metrics() {
    echo -e "${BLUE}=== Training Metrics ===${NC}"
    if [ -f "models/training_metrics.json" ]; then
        echo -e "${GREEN}File: models/training_metrics.json${NC}"
        echo ""
        cat models/training_metrics.json | jq '.'
    else
        echo -e "${RED}No training metrics found${NC}"
    fi
}

show_evaluation_metrics() {
    echo -e "${BLUE}=== Evaluation Metrics ===${NC}"
    if [ -f "metrics/evaluation_results.json" ]; then
        echo -e "${GREEN}File: metrics/evaluation_results.json${NC}"
        echo ""
        cat metrics/evaluation_results.json | jq '.'
    else
        echo -e "${RED}No evaluation metrics found${NC}"
    fi
}

list_log_files() {
    echo -e "${BLUE}=== All Log Files ===${NC}"
    if [ -d "$LOGS_DIR" ]; then
        echo ""
        ls -lh ${LOGS_DIR}/*.log 2>/dev/null | awk '{printf "%-50s %10s %s %s %s\n", $9, $5, $6, $7, $8}' || echo "No log files found"
        echo ""
        echo -e "${GREEN}Total size:${NC}"
        du -sh ${LOGS_DIR} 2>/dev/null || echo "0"
    else
        echo -e "${RED}Logs directory not found${NC}"
    fi
}

list_metric_files() {
    echo -e "${BLUE}=== All Metric Files ===${NC}"
    if [ -d "$METRICS_DIR" ]; then
        echo ""
        find ${METRICS_DIR} -type f | while read file; do
            size=$(du -h "$file" | cut -f1)
            echo -e "${GREEN}$file${NC} ($size)"
        done
        echo ""
        echo -e "${GREEN}Total size:${NC}"
        du -sh ${METRICS_DIR} 2>/dev/null || echo "0"
    else
        echo -e "${RED}Metrics directory not found${NC}"
    fi
}

tail_training_logs() {
    echo -e "${BLUE}=== Tailing Training Logs (Ctrl+C to stop) ===${NC}"
    LATEST=$(ls -t ${LOGS_DIR}/train_*.log 2>/dev/null | head -1)
    if [ -f "$LATEST" ]; then
        echo -e "${GREEN}Following: $LATEST${NC}"
        echo ""
        tail -f "$LATEST"
    else
        echo -e "${RED}No training logs found${NC}"
    fi
}

search_logs() {
    echo -n "Enter search term: "
    read search_term
    echo ""
    echo -e "${BLUE}=== Searching logs for: $search_term ===${NC}"
    if [ -d "$LOGS_DIR" ]; then
        grep -r "$search_term" ${LOGS_DIR}/*.log 2>/dev/null || echo "No matches found"
    else
        echo -e "${RED}Logs directory not found${NC}"
    fi
}

collect_artifacts() {
    echo -e "${BLUE}=== Collecting Artifacts ===${NC}"
    if [ -f "scripts/collect_artifacts.sh" ]; then
        ./scripts/collect_artifacts.sh
    else
        echo -e "${RED}Artifact collection script not found${NC}"
    fi
}

clean_old_logs() {
    echo -e "${YELLOW}=== Cleaning Old Logs (keeping last 5 days) ===${NC}"
    if [ -d "$LOGS_DIR" ]; then
        COUNT=$(find ${LOGS_DIR} -name "*.log" -type f -mtime +5 | wc -l)
        if [ "$COUNT" -gt 0 ]; then
            echo "Found $COUNT old log files"
            echo -n "Delete them? (y/n): "
            read confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                find ${LOGS_DIR} -name "*.log" -type f -mtime +5 -delete
                find ${LOGS_DIR} -name "*.jsonl" -type f -mtime +5 -delete
                echo -e "${GREEN}Old logs deleted${NC}"
            else
                echo "Cancelled"
            fi
        else
            echo -e "${GREEN}No old logs to delete${NC}"
        fi
    else
        echo -e "${RED}Logs directory not found${NC}"
    fi
}

# Main loop
while true; do
    show_menu
    read option
    
    case $option in
        1) view_latest_training_logs ;;
        2) view_latest_preprocess_logs ;;
        3) view_latest_evaluate_logs ;;
        4) view_error_logs ;;
        5) show_training_metrics ;;
        6) show_evaluation_metrics ;;
        7) list_log_files ;;
        8) list_metric_files ;;
        9) tail_training_logs ;;
        10) search_logs ;;
        11) collect_artifacts ;;
        12) clean_old_logs ;;
        0)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
    
    echo ""
    echo -n "Press Enter to continue..."
    read
done
