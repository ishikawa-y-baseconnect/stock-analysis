"""製薬パイプライン情報収集モジュール（ClinicalTrials.gov API使用）"""

from datetime import date, datetime
from typing import Any

import requests

from pharma_stock.config.companies import TOP_TIER_PHARMA_COMPANIES, PharmaCompany
from pharma_stock.storage.models import PipelineEvent

from .base import BaseCollector


class PipelineCollector(BaseCollector[PipelineEvent]):
    """ClinicalTrials.govからパイプライン情報を収集"""

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    # 企業名のマッピング（ClinicalTrials.gov検索用）
    COMPANY_SEARCH_NAMES: dict[str, list[str]] = {
        "4502.T": ["Takeda", "Takeda Pharmaceutical"],
        "4503.T": ["Astellas", "Astellas Pharma"],
        "4568.T": ["Daiichi Sankyo", "Daiichi-Sankyo"],
        "4523.T": ["Eisai"],
        "4519.T": ["Chugai", "Chugai Pharmaceutical"],
        "4578.T": ["Otsuka", "Otsuka Pharmaceutical"],
        "4506.T": ["Sumitomo Pharma", "Sumitomo Dainippon"],
        "4507.T": ["Shionogi"],
        "4151.T": ["Kyowa Kirin", "Kyowa Hakko Kirin"],
        "4528.T": ["Ono Pharmaceutical", "ONO PHARMACEUTICAL"],
    }

    def collect(
        self,
        tickers: list[str] | None = None,
        phase: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[PipelineEvent]:
        """パイプライン情報を収集

        Args:
            tickers: ティッカーシンボルのリスト
            phase: フェーズフィルタ（PHASE1, PHASE2, PHASE3, PHASE4）
            status: ステータスフィルタ（RECRUITING, COMPLETED等）
            limit: 取得件数上限

        Returns:
            パイプラインイベントのリスト
        """
        if tickers is None:
            tickers = [c.ticker for c in TOP_TIER_PHARMA_COMPANIES]

        self._log_start(tickers=tickers, phase=phase, status=status)

        results: list[PipelineEvent] = []

        for ticker in tickers:
            try:
                events = self._fetch_trials_for_company(ticker, phase, status, limit)
                results.extend(events)
            except Exception as e:
                self._log_error(e, ticker=ticker)
                continue

        self._log_complete(count=len(results))
        return results

    def _fetch_trials_for_company(
        self,
        ticker: str,
        phase: str | None,
        status: str | None,
        limit: int,
    ) -> list[PipelineEvent]:
        """特定企業の臨床試験データを取得"""
        search_names = self.COMPANY_SEARCH_NAMES.get(ticker, [])
        if not search_names:
            self.logger.warning("no_search_name", ticker=ticker)
            return []

        events: list[PipelineEvent] = []

        for search_name in search_names:
            params: dict[str, Any] = {
                "query.spons": search_name,
                "pageSize": min(limit, 100),
                "format": "json",
            }

            if phase:
                params["filter.phase"] = phase
            if status:
                params["filter.overallStatus"] = status

            try:
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.settings.request_timeout,
                )
                response.raise_for_status()
                data = response.json()

                studies = data.get("studies", [])
                for study in studies:
                    event = self._parse_study(ticker, study)
                    if event:
                        events.append(event)

            except requests.RequestException as e:
                self.logger.warning(
                    "api_request_failed",
                    search_name=search_name,
                    error=str(e),
                )
                continue

        return events

    def _parse_study(self, ticker: str, study: dict[str, Any]) -> PipelineEvent | None:
        """臨床試験データをパース"""
        try:
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            conditions = protocol.get("conditionsModule", {})

            nct_id = identification.get("nctId", "")
            title = identification.get("briefTitle", "")
            phases = design.get("phases", [])
            phase = phases[0] if phases else "N/A"
            overall_status = status_module.get("overallStatus", "")

            # 開始日を取得
            start_date_struct = status_module.get("startDateStruct", {})
            start_date_str = start_date_struct.get("date", "")

            # 日付パース
            event_date = self._parse_date(start_date_str)

            # 適応症
            condition_list = conditions.get("conditions", [])
            indication = ", ".join(condition_list[:3]) if condition_list else "N/A"

            return PipelineEvent(
                ticker=ticker,
                drug_name=title,
                indication=indication,
                phase=self._normalize_phase(phase),
                event_type=self._status_to_event_type(overall_status),
                event_date=event_date,
                source_url=f"https://clinicaltrials.gov/study/{nct_id}",
                details={
                    "nct_id": nct_id,
                    "overall_status": overall_status,
                    "phases": phases,
                },
            )
        except Exception as e:
            self.logger.warning("parse_study_failed", error=str(e))
            return None

    def _parse_date(self, date_str: str) -> date:
        """日付文字列をパース"""
        if not date_str:
            return date.today()

        # 様々なフォーマットに対応
        formats = ["%Y-%m-%d", "%Y-%m", "%B %Y", "%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        return date.today()

    def _normalize_phase(self, phase: str) -> str:
        """フェーズ名を正規化"""
        phase_map = {
            "PHASE1": "Phase 1",
            "PHASE2": "Phase 2",
            "PHASE3": "Phase 3",
            "PHASE4": "Phase 4",
            "EARLY_PHASE1": "前臨床",
            "NA": "N/A",
        }
        return phase_map.get(phase.upper().replace(" ", ""), phase)

    def _status_to_event_type(self, status: str) -> str:
        """ステータスをイベントタイプに変換"""
        status_map = {
            "RECRUITING": "募集中",
            "ACTIVE_NOT_RECRUITING": "進行中",
            "COMPLETED": "完了",
            "TERMINATED": "中止",
            "SUSPENDED": "中断",
            "WITHDRAWN": "撤回",
            "NOT_YET_RECRUITING": "準備中",
        }
        return status_map.get(status.upper().replace(",", "_").replace(" ", "_"), status)

    def get_phase3_trials(self, ticker: str | None = None) -> list[PipelineEvent]:
        """Phase 3試験を取得（最も株価に影響しやすい）"""
        tickers = [ticker] if ticker else None
        return self.collect(tickers=tickers, phase="PHASE3")

    def get_active_trials(self, ticker: str | None = None) -> list[PipelineEvent]:
        """進行中の試験を取得"""
        tickers = [ticker] if ticker else None
        return self.collect(tickers=tickers, status="RECRUITING")
