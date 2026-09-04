import { firebaseService, db, auth } from './firebaseService.js';
import { API_BASE } from './constants.js';
import { createInitialState } from './createInitialState.js';
import { getDemoLang, setDemoLang, toggleDemoLang } from './demoLang.js';
import { MapController } from './MapController.js';
import { SimulationController } from './SimulationController.js';
import { TelemetryService } from './TelemetryService.js';
import { solveDemo } from './DemoEngine.js';
import { FleetController } from './FleetController.js';
import { SettingsController } from './SettingsController.js';
import { GanttController } from './GanttController.js';
import { createSkillBadge, SKILL_CONFIG } from './skillUtils.js';

const APP_COPY = {
  en: {
    langButton: 'VN',
    backToDemo: 'Demo',
    authTitle: 'VRPTW Dispatch Portal',
    loginButton: 'Login',
    forgotPassword: 'Forgot Password?',
    openRegister: 'Create Account',
    guestLogin: 'Continue as Guest',
    guestHint: 'Demo mode active - skip auth and try the VRPTW solver right away.',
    requestOtp: 'Send OTP',
    registerButton: 'Register',
    backToLogin: 'Come Back',
    forgotButton: 'Send Reset Link',
    resetButton: 'Update Password',
    logoutButton: 'Logout',
    feedbackLink: 'Feedback',
    sidebarSubtitle: 'Ops Control',
    configTitle: 'Configuration',
    configNote: 'Fleet, data, import.',
    modeLabel: 'Data Mode',
    datasetLabel: 'Solomon Instance',
    importLabel: 'Import from Excel / CSV',
    dropzoneTitle: 'Drag and drop a file here',
    dropzoneCopy: '.xlsx, .xls, .csv or paste data directly.',
    sessionLabel: 'Session',
    heroKicker: 'Executive Control Panel',
    heroTitle: 'Real-time AI-powered delivery dispatch.',
    heroCopy: 'Focus on core data and operational decisions.',
    demoKicker: 'VRPTW Inference Demo',
    customerTitle: 'Customer List',
    customerNote:
      'Each row has a time window (Ready / Due) and Service duration. Units are consistent with distance (~1 km ≈ 1 time unit).',
    tabs: ['Overview', 'Split Map', 'Results', 'Archive'],
    helpTitle: 'How to drive the VRPTW demo',
    helpSubtitle: 'Five steps from sample data to a side-by-side DDQN vs. ALNS comparison.',
    helpClose: 'Close',
    helpSteps: [
      [
        'Pick a data mode',
        'Toggle Solomon RC (built-in demo instance, 12 customers around HCMC) or Real Data to upload your own.',
      ],
      [
        'Edit the table inline',
        'Double-click any cell to update name, lat/lng, demand, or the time window (ready / due / service). Add rows with + or import from .csv/.xlsx.',
      ],
      [
        'Tune the fleet',
        'Use the sliders for vehicles and capacity. The first row (id 0) is the depot and is non-editable.',
      ],
      [
        'Run Model',
        'Click Run Model. The backend solves twice: DDQN-ALNS and the ALNS baseline. Progress streams to the loading panel - you can minimize it.',
      ],
      [
        'Read the comparison',
        'Switch to the Maps tab for routes side-by-side, or the Results tab for runtime, total distance, vehicles used, and fleet utilisation deltas.',
      ],
    ],
    helpTips: [
      'Click the map to drop a customer pin (Real Data mode only).',
      'Paste a TSV/CSV anywhere on the page to import in Real Data mode.',
      'Demo / Guest mode: when Firebase is not configured, your data lives in memory only.',
      'Shortcut: press Shift + ? to reopen this guide.',
    ],
    analysisHeadings: [
      'Convergence Snapshot',
      'Operator Policy Heatmap',
      'Hourly Activity',
      'Instance Leaderboard',
      'Transfer Snapshot',
    ],
    analysisStatus: 'Analysis data is loading...',
    analysisLoadStatus: 'Loading analysis for',
    analysisReady: 'Analysis ready. Open Deep Analysis for full diagnostics.',
    analysisNoData: 'No activity data yet.',
    adminTitle: 'User Management',
    adminKicker: 'Admin',
    adminFeedbackKicker: 'Feedback Inbox',
    adminFeedbackTitle: 'Anonymous Feedback',
    adminRefresh: 'Refresh',
    adminFeedbackRefresh: 'Refresh',
    adminFeedbackLink: 'Feedback Inbox',
    loadingTitle: 'AI is optimizing routes...',
    loadingSubtitle: 'Sending request to the backend solver...',
    loadingPhase: 'Collecting route data...',
    loadingElapsed: 'Elapsed',
    loadingBackend: 'Backend',
    loadingCompute: 'Compute',
    loadingLauncher: 'Routes',
  },
  vn: {
    langButton: 'EN',
    backToDemo: 'Demo',
    authTitle: 'Cổng điều phối VRPTW',
    loginButton: 'Đăng nhập',
    forgotPassword: 'Quên mật khẩu?',
    openRegister: 'Tạo tài khoản',
    guestLogin: 'Vào với tư cách khách',
    guestHint: 'Đang ở chế độ demo - bỏ qua đăng nhập và thử solver VRPTW ngay.',
    requestOtp: 'Gửi OTP',
    registerButton: 'Đăng ký',
    backToLogin: 'Quay lại',
    forgotButton: 'Gửi link đặt lại',
    resetButton: 'Cập nhật mật khẩu',
    logoutButton: 'Đăng xuất',
    feedbackLink: 'Phản hồi',
    sidebarSubtitle: 'Điều khiển vận hành',
    configTitle: 'Cấu hình',
    configNote: 'Đội xe, dữ liệu, nhập liệu.',
    modeLabel: 'Chế độ dữ liệu',
    datasetLabel: 'Mẫu Solomon',
    importLabel: 'Nhập từ Excel / CSV',
    dropzoneTitle: 'Kéo thả file vào đây',
    dropzoneCopy: '.xlsx, .xls, .csv hoặc dán dữ liệu trực tiếp.',
    sessionLabel: 'Phiên làm việc',
    heroKicker: 'Bảng điều khiển điều hành',
    heroTitle: 'Điều phối giao hàng AI theo thời gian thực.',
    heroCopy: 'Tập trung vào dữ liệu lõi và quyết định vận hành.',
    demoKicker: 'Demo suy luận VRPTW',
    customerTitle: 'Danh sách khách hàng',
    customerNote:
      'Mỗi dòng có khung giờ (Ready / Due) và thời lượng Service. Đơn vị đồng nhất với khoảng cách (~1 km ≈ 1 đơn vị thời gian).',
    tabs: ['Tổng quan', 'Bản đồ tách đôi', 'Kết quả', 'Lưu trữ'],
    helpTitle: 'Cách dùng demo VRPTW',
    helpSubtitle: '5 bước từ dữ liệu mẫu tới so sánh DDQN và ALNS.',
    helpClose: 'Đóng',
    helpSteps: [
      [
        'Chọn chế độ dữ liệu',
        'Chuyển giữa Solomon RC (demo có sẵn, 12 khách ở HCMC) hoặc Real Data để tải dữ liệu riêng.',
      ],
      [
        'Sửa bảng trực tiếp',
        'Nháy đúp bất kỳ ô nào để sửa name, lat/lng, demand hoặc khung giờ (ready / due / service). Có thể thêm dòng bằng + hoặc import .csv/.xlsx.',
      ],
      [
        'Tinh chỉnh đội xe',
        'Dùng thanh trượt cho vehicles và capacity. Dòng đầu tiên (id 0) là depot và không được sửa.',
      ],
      [
        'Chạy mô hình',
        'Bấm Run Model. Backend chạy hai lần: DDQN-ALNS và ALNS baseline. Tiến trình hiển thị ở khung loading - có thể thu nhỏ.',
      ],
      [
        'Đọc phần so sánh',
        'Chuyển sang tab Maps để xem tuyến song song, hoặc tab Results để xem runtime, tổng quãng đường, số xe và chênh lệch tải.',
      ],
    ],
    helpTips: [
      'Bấm vào bản đồ để thả pin khách hàng (chỉ ở chế độ Real Data).',
      'Dán TSV/CSV ở bất kỳ đâu trên trang để import khi đang ở Real Data.',
      'Demo / Guest mode: khi chưa cấu hình Firebase, dữ liệu chỉ nằm trong bộ nhớ.',
      'Phím tắt: bấm Shift + ? để mở lại hướng dẫn này.',
    ],
    analysisHeadings: [
      'Ảnh chụp hội tụ',
      'Heatmap chính sách operator',
      'Hoạt động theo giờ',
      'Bảng xếp hạng instance',
      'Tổng quan transfer',
    ],
    analysisStatus: 'Đang tải dữ liệu phân tích...',
    analysisLoadStatus: 'Đang tải phân tích cho',
    analysisReady: 'Phân tích sẵn sàng. Mở Deep Analysis để xem đầy đủ.',
    analysisNoData: 'Chưa có dữ liệu hoạt động theo giờ.',
    adminTitle: 'Quản lý người dùng',
    adminKicker: 'Quản trị',
    adminFeedbackKicker: 'Hộp thư phản hồi',
    adminFeedbackTitle: 'Feedback ẩn danh',
    adminRefresh: 'Làm mới',
    adminFeedbackRefresh: 'Làm mới',
    adminFeedbackLink: 'Hộp thư phản hồi',
    loadingTitle: 'AI đang tối ưu tuyến...',
    loadingSubtitle: 'Đang gửi yêu cầu tới backend solver...',
    loadingPhase: 'Đang thu thập dữ liệu tuyến...',
    loadingElapsed: 'Đã chạy',
    loadingBackend: 'Backend',
    loadingCompute: 'Tính toán',
    loadingLauncher: 'Tuyến',
  },
};

export class App {
  constructor() {
    window.app = this;
    this.state = createInitialState();
    this.lang = getDemoLang();
    this.tableInputVisible = false;
    this.tableInputDraft = {
      name: '',
      address: '',
      demand: '0',
      lat: null,
      lng: null,
      ready: '0',
      due: '1000',
      service: '10',
    };
    this.selectedCustomerIds = new Set();
    this.tableInputRefs = {};
    this.tableAddressSuggest = [];
    this.tableAddressSuggestActive = -1;
    this.tableAddressSuggestTimer = 0;

    this.el = this.bindElements();
    this.mapController = new MapController(this);
    this.simulationController = new SimulationController(this);
    this.fleetController = new FleetController(this);
    this.settingsController = new SettingsController(this);
    this.ganttController = new GanttController(this);
    this.telemetry = new TelemetryService(db, this.mapController);
    this.loadingAnim = {
      active: false,
      rafId: 0,
      progress: 0,
      stage: 0,
      stageStartedAt: 0,
      lastTickAt: 0,
    };
    this.runSession = {
      token: 0,
      cancelled: false,
      abortController: null,
    };
    this.registerOtpCountdownTimer = 0;
    this.registerSuccessCountdownTimer = 0;
    this.registerOtpVerifyDebounceTimer = 0;
    this.isSendingRegisterOtp = false;
    this.isSendingForgotPasswordLink = false;
    this.isSubmittingPasswordChange = false;
    this.registerOtpRequestedEmail = ''; // Track which email OTP was sent to
    this.backendMode = { firebase_enabled: null, demo_mode: null, torch: null };
    this.state.lang = this.lang;
    this.applyLanguage(this.lang);
    this.wireAuthEvents();
    this.wireLoadingControls();
    this.wireLanguageControls();
    this.routeAuthScreenFromURL();

    this.probeBackendMode();

    if (this.state.unlocked && this.state.email) {
      if (this.state.mustChangePassword) {
        this.showAuthView('reset');
        this.toast('Password Change Required', 'Set a new password before accessing the app.', 'error');
      } else if (this.state.role === 'guest') {
        this.enterApp();
        this.toast('Demo Mode', 'Continuing as guest from previous session.', 'ok');
      } else {
        this.enterApp();
        this.initFirebase(this.state.email);
        this.toast('Auto Login', 'Previous session restored.', 'ok');
      }
    } else {
      this.leaveApp();
    }
  }

  bindElements() {
    return {
      viewForgot: document.getElementById('auth-view-forgot'),
      viewReset: document.getElementById('auth-view-reset'),
      viewLogin: document.getElementById('auth-view-login'),
      viewRegister: document.getElementById('auth-view-register'),
      authViews: Array.from(document.querySelectorAll('.auth-view')),
      loginEmail: document.getElementById('login-email'),
      loginPassword: document.getElementById('login-password'),
      registerEmail: document.getElementById('register-email'),
      registerPassword: document.getElementById('register-password'),
      registerOtp: document.getElementById('register-otp'),
      registerOtpCountdown: document.getElementById('register-otp-countdown'),
      forgotEmail: document.getElementById('forgot-email'),
      resetPassword: document.getElementById('reset-password'),
      resetPasswordConfirm: document.getElementById('reset-password-confirm'),
      linkForgotPassword: document.getElementById('link-forgot-password'),
      btnOpenRegister: document.getElementById('btn-open-register'),
      btnBackLoginFromRegister: document.getElementById('btn-back-login-from-register'),
      btnBackLoginFromForgot: document.getElementById('btn-back-login-from-forgot'),
      btnBackLoginFromReset: document.getElementById('btn-back-login-from-reset'),
      authHint: document.getElementById('auth-hint'),
      btnRequestOtp: document.getElementById('btn-request-otp'),
      btnRegister: document.getElementById('btn-register'),
      btnLogin: document.getElementById('btn-login'),
      btnGuestLogin: document.getElementById('btn-guest-login'),
      guestBlock: document.getElementById('guest-block'),
      guestHint: document.getElementById('guest-hint'),
      btnForgotPassword: document.getElementById('btn-forgot-password'),
      btnResetPassword: document.getElementById('btn-reset-password'),
      btnLogout: document.getElementById('btn-logout'),
      helpButton: document.getElementById('help-button'),
      backToLanding: document.querySelector('.back-to-landing'),
      feedbackLink: document.getElementById('feedback-link'),
      feedbackAdminLink: document.getElementById('feedback-admin-link'),
      langToggle: document.getElementById('lang-toggle'),
      adminPanel: document.getElementById('admin-panel'),
      adminRefresh: document.getElementById('admin-refresh'),
      adminFeedbackRefresh: document.getElementById('admin-feedback-refresh'),
      adminUserRows: document.getElementById('admin-user-rows'),
      adminFeedbackRows: document.getElementById('admin-feedback-rows'),
      userEmail: document.getElementById('user-email'),
      authScreen: document.getElementById('auth-screen'),
      appShell: document.getElementById('app-shell'),
      tabButtons: Array.from(document.querySelectorAll('.tab-btn')),
      tabPanels: Array.from(document.querySelectorAll('.tab-panel')),
      tabbarIndicator: document.querySelector('.tabbar-indicator'),
      pickExcel: document.getElementById('pick-excel'),
      excelInput: document.getElementById('excel-input'),
      dropzone: document.getElementById('dropzone'),
      modeToggle: document.getElementById('mode-toggle'),
      datasetPicker: document.getElementById('dataset-picker'),
      datasetSelect: document.getElementById('dataset-select'),
      loadDataset: document.getElementById('load-dataset'),
      vehicles: document.getElementById('vehicles-slider'),
      vehiclesValue: document.getElementById('vehicles-value'),
      capacity: document.getElementById('capacity-slider'),
      capacityValue: document.getElementById('capacity-value'),
      addressInput: document.getElementById('address-input'),
      addAddress: document.getElementById('add-address'),
      suggestList: document.getElementById('address-suggest'),
      pasteBox: document.getElementById('paste-box'),
      parsePaste: document.getElementById('parse-paste'),
      addRow: document.getElementById('add-row'),
      deleteSelected: document.getElementById('delete-selected'),
      runModel: document.getElementById('run-model'),
      customerRows: document.getElementById('customer-rows'),
      tableEmpty: document.getElementById('table-empty'),
      tableSkeleton: document.getElementById('table-skeleton'),
      mapEmptyDdqn: document.getElementById('map-empty-ddqn'),
      mapEmptyAlns: document.getElementById('map-empty-alns'),
      metricRuntimeCard: document.getElementById('metric-runtime-card'),
      metricRuntimeDdqn: document.getElementById('metric-runtime-ddqn'),
      metricRuntimeAlns: document.getElementById('metric-runtime-alns'),
      metricRuntimeDelta: document.getElementById('metric-runtime-delta'),
      metricRuntimeBarDdqn: document.getElementById('metric-runtime-bar-ddqn'),
      metricRuntimeBarAlns: document.getElementById('metric-runtime-bar-alns'),
      metricDistanceCard: document.getElementById('metric-distance-card'),
      metricDistanceDdqn: document.getElementById('metric-distance-ddqn'),
      metricDistanceAlns: document.getElementById('metric-distance-alns'),
      metricDistanceDelta: document.getElementById('metric-distance-delta'),
      metricDistanceBarDdqn: document.getElementById('metric-distance-bar-ddqn'),
      metricDistanceBarAlns: document.getElementById('metric-distance-bar-alns'),
      metricVehiclesCard: document.getElementById('metric-vehicles-card'),
      metricVehiclesDdqn: document.getElementById('metric-vehicles-ddqn'),
      metricVehiclesAlns: document.getElementById('metric-vehicles-alns'),
      metricVehiclesDelta: document.getElementById('metric-vehicles-delta'),
      metricVehiclesBarDdqn: document.getElementById('metric-vehicles-bar-ddqn'),
      metricVehiclesBarAlns: document.getElementById('metric-vehicles-bar-alns'),
      metricLoadCard: document.getElementById('metric-load-card'),
      metricLoadDdqn: document.getElementById('metric-load-ddqn'),
      metricLoadAlns: document.getElementById('metric-load-alns'),
      metricLoadDelta: document.getElementById('metric-load-delta'),
      metricLoadBarDdqn: document.getElementById('metric-load-bar-ddqn'),
      metricLoadBarAlns: document.getElementById('metric-load-bar-alns'),
      metricLoadDdqnState: document.getElementById('metric-load-ddqn-state'),
      metricLoadAlnsState: document.getElementById('metric-load-alns-state'),
      metricLoadDonutDdqn: document.getElementById('metric-load-donut-ddqn'),
      metricLoadDonutAlns: document.getElementById('metric-load-donut-alns'),
      metricLoadDonutDdqnLabel: document.getElementById('metric-load-donut-ddqn-label'),
      metricLoadDonutAlnsLabel: document.getElementById('metric-load-donut-alns-label'),
      analysisVersion: document.getElementById('analysis-version'),
      analysisInstance: document.getElementById('analysis-instance'),
      analysisOpenPopup: document.getElementById('analysis-open-popup'),
      analysisLastUpdated: document.getElementById('analysis-last-updated'),
      analysisStatus: document.getElementById('analysis-status'),
      analysisSummaryKpis: document.getElementById('analysis-summary-kpis'),
      analysisConvergenceChart: document.getElementById('analysis-convergence-chart'),
      analysisActivityChart: document.getElementById('analysis-activity-chart'),
      analysisPolicyGrid: document.getElementById('analysis-policy-grid'),
      analysisLeaderboardBody: document.getElementById('analysis-leaderboard-body'),
      analysisTransferBody: document.getElementById('analysis-transfer-body'),
      analysisModal: document.getElementById('analysis-modal'),
      analysisModalClose: document.getElementById('analysis-modal-close'),
      analysisModalSubtitle: document.getElementById('analysis-modal-subtitle'),
      analysisModalMeta: document.getElementById('analysis-modal-meta'),
      analysisModalConvergence: document.getElementById('analysis-modal-convergence'),
      analysisModalTransferPlot: document.getElementById('analysis-modal-transfer-plot'),
      analysisModalTransferBody: document.getElementById('analysis-modal-transfer-body'),
      connectionPill: document.getElementById('connection-pill'),
      loading: document.getElementById('loading'),
      loadingCard: document.getElementById('loading-card'),
      loadingMinimize: document.getElementById('loading-minimize'),
      loadingCancel: document.getElementById('loading-cancel'),
      loadingLauncher: document.getElementById('loading-launcher'),
      loadingLauncherText: document.querySelector('.loading-launcher-text'),
      loadingTitle: document.getElementById('loading-title'),
      loadingSubtitle: document.getElementById('loading-subtitle'),
      loadingPhase: document.getElementById('loading-phase'),
      loadingPercent: document.getElementById('loading-percent'),
      loadingTrackFill: document.getElementById('loading-track-fill'),
      loadingTruck: document.getElementById('loading-truck'),
      loadingElapsed: document.getElementById('loading-elapsed'),
      loadingBackend: document.getElementById('loading-backend'),
      loadingDevice: document.getElementById('loading-device'),
      loadingConsole: document.getElementById('loading-console'),
      toastRoot: document.getElementById('toast-root'),
      status: document.getElementById('status'),
    };
  }

  wireAuthEvents() {
    this.el.btnOpenRegister?.addEventListener('click', (event) => {
      event.preventDefault();
      this.showAuthView('register');
    });
    this.el.linkForgotPassword?.addEventListener('click', (event) => {
      event.preventDefault();
      this.showAuthView('forgot');
    });
    this.el.btnBackLoginFromRegister?.addEventListener('click', (event) => {
      event.preventDefault();
      this.showAuthView('login');
    });
    this.el.btnBackLoginFromForgot?.addEventListener('click', (event) => {
      event.preventDefault();
      this.showAuthView('login');
    });
    this.el.btnBackLoginFromReset?.addEventListener('click', (event) => {
      event.preventDefault();
      this.showAuthView('login');
    });
    this.el.btnRequestOtp?.addEventListener('click', (event) => {
      event.preventDefault();
      this.requestRegisterOtp();
    });
    this.el.btnRegister?.addEventListener('click', (event) => {
      event.preventDefault();
      this.register();
    });
    this.el.btnLogin?.addEventListener('click', (event) => {
      event.preventDefault();
      this.login();
    });
    this.el.btnGuestLogin?.addEventListener('click', (event) => {
      event.preventDefault();
      this.loginAsGuest();
    });
    this.el.btnForgotPassword?.addEventListener('click', (event) => {
      event.preventDefault();
      this.requestForgotPasswordOtp();
    });
    this.el.btnResetPassword?.addEventListener('click', (event) => {
      event.preventDefault();
      this.resetForgotPassword();
    });
    this.el.btnLogout?.addEventListener('click', (event) => {
      event.preventDefault();
      this.logout();
    });

    this.el.registerEmail?.addEventListener('input', () => {
      this.clearFieldError(this.el.registerEmail);
      const currentEmail = this.el.registerEmail.value.trim().toLowerCase();
      // Only reset OTP state if email actually changed from the one OTP was sent to
      if (currentEmail !== this.registerOtpRequestedEmail) {
        this.state.registerOtpApprovedEmail = '';
        this.state.registerOtpVerified = false;
        this.state.registerOtpExpiresAt = 0;
        this.stopRegisterOtpCountdown();
        this.updateRegisterOtpCountdownText('Click Send OTP to receive a verification code.');
        this.updateRegisterButtonState();
      }
    });
    this.el.registerPassword?.addEventListener('input', () => this.clearFieldError(this.el.registerPassword));
    this.el.registerOtp?.addEventListener('input', () => {
      this.clearFieldError(this.el.registerOtp);
      this.state.registerOtpVerified = false;
      this.updateRegisterButtonState();
      this.scheduleRealtimeOtpVerification();
    });
    [this.el.registerEmail, this.el.registerPassword, this.el.registerOtp].forEach((field) => {
      field?.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter') return;
        event.preventDefault();
      });
    });
    this.el.loginEmail?.addEventListener('input', () => this.clearFieldError(this.el.loginEmail));
    this.el.loginPassword?.addEventListener('input', () => this.clearFieldError(this.el.loginPassword));
    this.el.forgotEmail?.addEventListener('input', () => this.clearFieldError(this.el.forgotEmail));
    this.el.resetPassword?.addEventListener('input', () => this.clearFieldError(this.el.resetPassword));
    this.el.resetPasswordConfirm?.addEventListener('input', () => this.clearFieldError(this.el.resetPasswordConfirm));

    this.updateRegisterButtonState();
  }

  wireLoadingControls() {
    this.el.loadingMinimize?.addEventListener('click', (event) => {
      event.preventDefault();
      this.minimizeLoading();
    });
    this.el.loadingCancel?.addEventListener('click', (event) => {
      event.preventDefault();
      this.cancelLoading();
    });
    this.el.loadingLauncher?.addEventListener('click', (event) => {
      event.preventDefault();
      this.restoreLoading();
    });
  }

  wireLanguageControls() {
    this.el.langToggle?.addEventListener('click', () => {
      this.applyLanguage(toggleDemoLang());
    });
  }

  applyLanguage(lang) {
    const next = lang === 'vn' ? 'vn' : 'en';
    this.lang = next;
    this.state.lang = next;
    setDemoLang(next);

    const copy = APP_COPY[next];
    document.documentElement.lang = next === 'vn' ? 'vi' : 'en';

    const setText = (node, value) => {
      if (node) node.textContent = value;
    };

    setText(this.el.langToggle, copy.langButton);
    setText(document.getElementById('auth-title'), copy.authTitle);
    setText(this.el.btnLogin, copy.loginButton);
    setText(this.el.linkForgotPassword, copy.forgotPassword);
    setText(this.el.btnOpenRegister, copy.openRegister);
    setText(this.el.btnGuestLogin, copy.guestLogin);
    setText(this.el.guestHint, copy.guestHint);
    setText(this.el.btnRequestOtp, copy.requestOtp);
    setText(this.el.btnRegister, copy.registerButton);
    setText(this.el.btnBackLoginFromRegister, copy.backToLogin);
    setText(this.el.btnBackLoginFromForgot, copy.backToLogin);
    setText(this.el.btnBackLoginFromReset, copy.backToLogin);
    setText(this.el.btnForgotPassword, copy.forgotButton);
    setText(this.el.btnResetPassword, copy.resetButton);
    setText(this.el.btnLogout, copy.logoutButton);
    setText(this.el.feedbackLink, copy.feedbackLink);
    setText(this.el.feedbackAdminLink, copy.adminFeedbackLink);
    setText(document.getElementById('sidebar-subtitle'), copy.sidebarSubtitle);
    setText(document.getElementById('config-title'), copy.configTitle);
    setText(document.getElementById('config-note'), copy.configNote);
    setText(document.getElementById('mode-toggle-label'), copy.modeLabel);
    setText(document.getElementById('dataset-picker-label'), copy.datasetLabel);
    setText(document.getElementById('import-label'), copy.importLabel);
    setText(document.getElementById('dropzone-title'), copy.dropzoneTitle);
    setText(document.getElementById('dropzone-copy'), copy.dropzoneCopy);
    setText(document.getElementById('session-kicker'), copy.sessionLabel);
    setText(document.getElementById('hero-kicker'), copy.heroKicker);
    setText(document.getElementById('hero-title'), copy.heroTitle);
    setText(document.getElementById('hero-copy'), copy.heroCopy);
    setText(document.getElementById('demo-kicker'), copy.demoKicker);
    setText(document.getElementById('customer-title'), copy.customerTitle);
    setText(document.getElementById('customer-note'), copy.customerNote);
    setText(this.el.adminRefresh, copy.adminRefresh);
    setText(this.el.adminFeedbackRefresh, copy.adminFeedbackRefresh);
    setText(document.getElementById('admin-title'), copy.adminTitle);
    setText(document.getElementById('admin-kicker'), copy.adminKicker);
    setText(document.getElementById('admin-feedback-kicker'), copy.adminFeedbackKicker);
    setText(document.getElementById('admin-feedback-title'), copy.adminFeedbackTitle);

    this.el.tabButtons.forEach((button, index) => {
      if (copy.tabs[index]) button.textContent = copy.tabs[index];
    });

    const analysisTitles = document.querySelectorAll('.analysis-block h3');
    analysisTitles.forEach((node, index) => {
      if (copy.analysisHeadings[index]) node.textContent = copy.analysisHeadings[index];
    });

    const loadingLabels = document.querySelectorAll('.loading-stat-label');
    if (loadingLabels[0]) loadingLabels[0].textContent = copy.loadingElapsed;
    if (loadingLabels[1]) loadingLabels[1].textContent = copy.loadingBackend;
    if (loadingLabels[2]) loadingLabels[2].textContent = copy.loadingCompute;
    setText(this.el.loadingTitle, copy.loadingTitle);
    setText(this.el.loadingSubtitle, copy.loadingSubtitle);
    setText(this.el.loadingPhase, copy.loadingPhase);
    setText(this.el.loadingLauncherText, copy.loadingLauncher);

    setText(document.getElementById('help-modal-title'), copy.helpTitle);
    setText(document.querySelector('.help-modal-sub'), copy.helpSubtitle);
    setText(document.getElementById('help-modal-close'), copy.helpClose);

    const helpSteps = document.querySelectorAll('.help-steps li');
    helpSteps.forEach((item, index) => {
      const title = item.querySelector('h3');
      const body = item.querySelector('p');
      const step = copy.helpSteps[index];
      if (!step) return;
      if (title) title.textContent = step[0];
      if (body) body.textContent = step[1];
    });

    const tips = document.querySelectorAll('.help-tips li');
    tips.forEach((item, index) => {
      if (copy.helpTips[index]) item.textContent = copy.helpTips[index];
    });

    if (this.el.helpButton) {
      this.el.helpButton.title = next === 'vn' ? 'Hướng dẫn nhanh (Shift+?)' : 'Quick help (Shift+?)';
      this.el.helpButton.setAttribute('aria-label', next === 'vn' ? 'Mở hướng dẫn nhanh' : 'Open quick help');
    }

    if (this.el.langToggle) {
      this.el.langToggle.title = next === 'vn' ? 'Chuyển sang tiếng Anh' : 'Switch to Vietnamese';
      this.el.langToggle.setAttribute('aria-label', next === 'vn' ? 'Chuyển sang tiếng Anh' : 'Switch language');
    }

    if (this.el.backToLanding) {
      this.el.backToLanding.title = next === 'vn' ? 'Quay lại trang chủ' : 'Back to landing page';
    }

    if (this.el.status && this.el.status.textContent === 'Ready.') {
      this.el.status.textContent = next === 'vn' ? 'Sẵn sàng.' : 'Ready.';
    }

    this.applyBackendMode();
    this.renderActivityChart(this.el.analysisActivityChart, this.state.analysisActivity);
    if (this.state.adminFeedback) {
      this.renderAdminFeedback(this.state.adminFeedback);
    }

    if (this.state.currentView) {
      this.switchView(this.state.currentView);
    }
    this.fleetController?.render();
    this.settingsController?.render();
  }

  routeAuthScreenFromURL() {
    const params = new URLSearchParams(window.location.search);
    const screen = params.get('screen') || sessionStorage.getItem('vrptw_auth_screen');
    if (screen === 'register') {
      this.showAuthView('register');
      return;
    }
    if (screen === 'forgot') {
      this.showAuthView('forgot');
      return;
    }
    if (screen === 'reset') {
      this.state.resetToken = params.get('token') || '';
      this.showAuthView('reset');
      return;
    }
    this.showAuthView('login');
  }

  showAuthView(view) {
    const key = `view${view.charAt(0).toUpperCase()}${view.slice(1)}`;
    const current = this.el[key];
    this.el.authViews.forEach((node) => node.classList.add('hidden'));
    current?.classList.remove('hidden');
    if (view) sessionStorage.setItem('vrptw_auth_screen', view);
    this.syncAuthScreenInUrl(view);
    this.stopRegisterSuccessCountdown();
    this.clearAuthInputErrors();
    // Clear register OTP state when leaving register view
    if (view !== 'register') {
      this.registerOtpRequestedEmail = '';
    }
    if (view === 'register') {
      if (this.state.registerOtpExpiresAt > Date.now()) {
        this.startRegisterOtpCountdown();
      } else {
        this.stopRegisterOtpCountdown();
        this.updateRegisterOtpCountdownText(
          this.lang === 'vn' ? 'Bấm Gửi OTP để nhận mã xác thực.' : 'Click Send OTP to receive a verification code.'
        );
      }
      this.updateRegisterButtonState();
    } else {
      this.stopRegisterOtpCountdown();
    }
  }

  syncAuthScreenInUrl(view) {
    const url = new URL(window.location.href);
    if (view === 'register' || view === 'forgot') {
      url.searchParams.set('screen', view);
      if (view !== 'reset') url.searchParams.delete('token');
    } else if (view === 'reset') {
      url.searchParams.set('screen', 'reset');
      if (this.state.resetToken) url.searchParams.set('token', this.state.resetToken);
    } else {
      url.searchParams.delete('screen');
      url.searchParams.delete('token');
    }
    window.history.replaceState({}, '', `${url.pathname}${url.search}`);
  }

  clearAuthInputErrors() {
    [
      this.el.loginEmail,
      this.el.loginPassword,
      this.el.registerEmail,
      this.el.registerPassword,
      this.el.registerOtp,
      this.el.forgotEmail,
      this.el.resetPassword,
      this.el.resetPasswordConfirm,
    ].forEach((field) => this.clearFieldError(field));
    if (this.el.authHint) this.el.authHint.style.display = 'none';
  }

  setFieldError(field) {
    if (!field) return;
    field.classList.add('input-error');
  }

  clearFieldError(field) {
    if (!field) return;
    field.classList.remove('input-error');
  }

  setStatus(message, tone = '') {
    if (!this.el.status) return;
    this.el.status.textContent = message || '';
    this.el.status.className = `status${tone ? ` ${tone}` : ''}`;
  }

  toast(title, message = '', tone = '') {
    if (!this.el.toastRoot) return;

    const node = document.createElement('div');
    node.className = `toast${tone ? ` ${tone}` : ''}`;
    node.setAttribute('role', tone === 'error' ? 'alert' : 'status');

    const titleNode = document.createElement('div');
    titleNode.className = 'toast-title';
    titleNode.textContent = title || 'Notice';
    node.appendChild(titleNode);

    if (message) {
      const messageNode = document.createElement('div');
      messageNode.className = 'toast-message';
      messageNode.textContent = message;
      node.appendChild(messageNode);
    }

    this.el.toastRoot.appendChild(node);
    window.setTimeout(() => node.remove(), tone === 'error' ? 6500 : 4200);
  }

  updateRegisterButtonState() {
    if (!this.el.btnRegister || !this.el.registerEmail) return;
    const currentEmail = this.el.registerEmail.value.trim().toLowerCase();
    const enabled = Boolean(
      this.state.registerOtpApprovedEmail &&
      this.state.registerOtpApprovedEmail === currentEmail &&
      this.state.registerOtpVerified &&
      this.state.registerOtpExpiresAt > Date.now()
    );

    this.el.btnRegister.hidden = false;
    this.el.btnRegister.classList.remove('hidden');
    this.el.btnRegister.disabled = !enabled;
  }

  stopRegisterOtpCountdown() {
    if (this.registerOtpCountdownTimer) {
      window.clearInterval(this.registerOtpCountdownTimer);
      this.registerOtpCountdownTimer = 0;
    }
  }

  stopRegisterSuccessCountdown() {
    if (this.registerSuccessCountdownTimer) {
      window.clearInterval(this.registerSuccessCountdownTimer);
      this.registerSuccessCountdownTimer = 0;
    }
  }

  stopRegisterOtpVerifyDebounce() {
    if (this.registerOtpVerifyDebounceTimer) {
      window.clearTimeout(this.registerOtpVerifyDebounceTimer);
      this.registerOtpVerifyDebounceTimer = 0;
    }
  }

  scheduleRealtimeOtpVerification() {
    this.stopRegisterOtpVerifyDebounce();

    const email = this.el.registerEmail?.value.trim().toLowerCase() || '';
    const otp = this.el.registerOtp?.value.trim() || '';
    const canVerify =
      this.state.registerOtpApprovedEmail === email &&
      this.state.registerOtpExpiresAt > Date.now() &&
      /^\d{6}$/.test(otp);

    if (!canVerify) return;

    this.registerOtpVerifyDebounceTimer = window.setTimeout(() => {
      this.verifyRegisterOtp({ silent: true });
    }, 220);
  }

  updateRegisterOtpCountdownText(text, tone = '') {
    if (!this.el.registerOtpCountdown) return;
    this.el.registerOtpCountdown.className = `otp-countdown ${tone}`.trim();
    this.el.registerOtpCountdown.textContent = text;
  }

  startRegisterOtpCountdown() {
    this.stopRegisterOtpCountdown();

    const tick = () => {
      const remainMs = this.state.registerOtpExpiresAt - Date.now();
      if (remainMs <= 0) {
        this.stopRegisterOtpCountdown();
        this.state.registerOtpApprovedEmail = '';
        this.state.registerOtpVerified = false;
        this.state.registerOtpExpiresAt = 0;
        this.updateRegisterButtonState();
        this.updateRegisterOtpCountdownText('OTP expired. Please click Send OTP again.', 'expired');
        return;
      }

      const remainSec = Math.ceil(remainMs / 1000);
      const minutes = Math.floor(remainSec / 60);
      const seconds = remainSec % 60;
      this.updateRegisterOtpCountdownText(`OTP valid for ${minutes}:${String(seconds).padStart(2, '0')}.`, 'active');
      this.updateRegisterButtonState();
    };

    tick();
    this.registerOtpCountdownTimer = window.setInterval(tick, 1000);
  }

  parseApiError(error) {
    const raw = String(error?.message || '').trim();
    if (!raw) return 'An error occurred';
    try {
      const parsed = JSON.parse(raw);
      return parsed.detail || parsed.message || raw;
    } catch {
      return raw;
    }
  }

  autoAdjustVehiclesForInfeasible(message) {
    const text = String(message || '').trim();
    if (!text) return null;

    const lower = text.toLowerCase();
    const looksInfeasible =
      (lower.includes('infeasible configuration') && lower.includes('vehicles')) ||
      (lower.includes('infeasible with current settings') && lower.includes('vehicles'));
    if (!looksInfeasible) return null;

    let recommended = null;
    const reqMatch = text.match(/requires\s+(\d+)\s+vehicles?\s+but only\s+(\d+)/i);
    if (reqMatch) {
      recommended = Number(reqMatch[1]);
    } else {
      const capacity = Math.max(1, Number(this.state.capacity) || 1);
      const totalDemand = this.state.customers
        .filter((c) => !c.isDepot)
        .reduce((sum, c) => sum + Math.max(0, Number(c.demand) || 0), 0);
      recommended = Math.max(1, Math.ceil(totalDemand / capacity));
    }

    if (!Number.isFinite(recommended) || recommended <= 0) return null;

    const slider = this.el.vehicles;
    const sliderMax = Math.max(1, Number(slider?.max) || 30);
    const current = Math.max(1, Number(this.state.vehicles) || 1);
    const target = Math.min(sliderMax, Math.max(current, Math.ceil(recommended)));
    if (target <= current) return { changed: false, target, current, max: sliderMax };

    this.state.vehicles = target;
    if (slider) slider.value = String(target);
    if (this.el.vehiclesValue) this.el.vehiclesValue.textContent = String(target);

    return { changed: true, target, current, max: sliderMax };
  }

  formatRunError(errorLike) {
    const raw = typeof errorLike === 'string' ? errorLike : this.parseApiError(errorLike);
    const message = String(raw || '').trim();
    if (!message) return 'Run failed unexpectedly.';

    const lower = message.toLowerCase();
    if (lower.includes('infeasible configuration') && lower.includes('vehicles')) {
      const reqMatch = message.match(/requires\s+(\d+)\s+vehicles?\s+but only\s+(\d+)/i);
      if (reqMatch) {
        const needed = Number(reqMatch[1]);
        const current = Number(reqMatch[2]);
        return `Infeasible: need at least ${needed} vehicles (current: ${current}). Increase Vehicles or Capacity.`;
      }
      const capacity = Math.max(1, Number(this.state.capacity) || 1);
      const totalDemand = this.state.customers
        .filter((c) => !c.isDepot)
        .reduce((sum, c) => sum + Math.max(0, Number(c.demand) || 0), 0);
      const minVehiclesByDemand = Math.max(1, Math.ceil(totalDemand / capacity));
      const currentVehicles = Math.max(1, Number(this.state.vehicles) || 1);
      const recommendation = Math.max(minVehiclesByDemand, currentVehicles + 1);
      return `Infeasible with current settings. Try Vehicles >= ${recommendation} (current: ${currentVehicles}) or increase Capacity.`;
    }
    if (lower.includes('infeasible with current settings') && lower.includes('vehicles')) {
      const capacity = Math.max(1, Number(this.state.capacity) || 1);
      const totalDemand = this.state.customers
        .filter((c) => !c.isDepot)
        .reduce((sum, c) => sum + Math.max(0, Number(c.demand) || 0), 0);
      const minVehiclesByDemand = Math.max(1, Math.ceil(totalDemand / capacity));
      const currentVehicles = Math.max(1, Number(this.state.vehicles) || 1);
      const recommendation = Math.max(minVehiclesByDemand, currentVehicles + 1);
      return `Infeasible with current settings. Try Vehicles >= ${recommendation} (current: ${currentVehicles}) or increase Capacity.`;
    }
    if (lower.includes('exceeds vehicle capacity')) {
      return 'At least one customer demand exceeds vehicle capacity. Increase Capacity or reduce that customer demand.';
    }
    if (lower.includes('negative demand')) {
      return 'Invalid input: demand cannot be negative.';
    }
    return message;
  }

  isValidEmail(email) {
    return /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/.test(email);
  }

  async requestRegisterOtp() {
    if (this.isSendingRegisterOtp) return;

    // Keep the register view pinned while sending OTP.
    this.showAuthView('register');

    const email = this.el.registerEmail.value.trim().toLowerCase();
    this.clearFieldError(this.el.registerEmail);
    this.clearFieldError(this.el.registerOtp);

    if (!email) {
      this.setFieldError(this.el.registerEmail);
      this.updateRegisterOtpCountdownText('Email is required before sending OTP.', 'expired');
      this.toast('Missing Email', 'Please enter your email first.', 'error');
      return;
    }

    if (!this.isValidEmail(email)) {
      this.setFieldError(this.el.registerEmail);
      this.updateRegisterOtpCountdownText('Invalid email format.', 'expired');
      this.toast('Invalid Email', 'Please enter a valid email before sending OTP.', 'error');
      return;
    }

    try {
      this.isSendingRegisterOtp = true;
      this.el.btnRequestOtp && (this.el.btnRequestOtp.disabled = true);
      this.updateRegisterOtpCountdownText('Sending OTP...', 'active');

      const res = await this.request('/auth/register/request-otp', {
        method: 'POST',
        body: JSON.stringify({ email }),
      });

      this.state.registerOtpApprovedEmail = email;
      this.registerOtpRequestedEmail = email; // Track this email for OTP state management
      this.state.registerOtpVerified = false;
      this.state.registerOtpExpiresAt = Date.now() + 10 * 60 * 1000;
      this.stopRegisterOtpVerifyDebounce();
      this.startRegisterOtpCountdown();
      this.updateRegisterButtonState();
      this.el.registerOtp?.focus();
      this.toast('OTP Sent Successfully', `Delivery method: ${res.delivery}. Check your email for the OTP code.`, 'ok');
      this.updateRegisterOtpCountdownText('OTP sent. Enter OTP to verify automatically.', 'active');
      this.setStatus('Send OTP success. Please enter the 6-digit OTP to continue.', 'ok');
      // Pin register view after successful OTP send
      this.showAuthView('register');
    } catch (error) {
      this.state.registerOtpApprovedEmail = '';
      this.registerOtpRequestedEmail = ''; // Clear on error
      this.state.registerOtpVerified = false;
      this.state.registerOtpExpiresAt = 0;
      this.stopRegisterOtpVerifyDebounce();
      this.stopRegisterOtpCountdown();
      this.updateRegisterOtpCountdownText('Failed to send OTP. Please try again.', 'expired');
      this.updateRegisterButtonState();
      this.setFieldError(this.el.registerEmail);
      const reason = this.parseApiError(error);
      this.setStatus(`Send OTP failed: ${reason}`, 'error');
      this.toast('Failed to Send OTP', reason, 'error');
      this.showAuthView('register');
    } finally {
      this.isSendingRegisterOtp = false;
      this.el.btnRequestOtp && (this.el.btnRequestOtp.disabled = false);
    }
  }

  async verifyRegisterOtp(options = {}) {
    const { silent = false } = options;
    try {
      const email = this.el.registerEmail.value.trim().toLowerCase();
      const otp = this.el.registerOtp.value.trim();

      this.clearFieldError(this.el.registerEmail);
      this.clearFieldError(this.el.registerOtp);

      if (!email || !this.isValidEmail(email)) {
        this.setFieldError(this.el.registerEmail);
        throw new Error('Invalid email format');
      }
      if (!otp) {
        this.setFieldError(this.el.registerOtp);
        throw new Error('OTP is required');
      }
      if (!/^\d{6}$/.test(otp)) {
        this.setFieldError(this.el.registerOtp);
        throw new Error('OTP must be exactly 6 digits');
      }
      if (this.state.registerOtpApprovedEmail !== email || this.state.registerOtpExpiresAt <= Date.now()) {
        this.setFieldError(this.el.registerEmail);
        throw new Error('Please send OTP first before verifying');
      }

      await this.request('/auth/register/verify-otp', {
        method: 'POST',
        body: JSON.stringify({ email, otp }),
      });

      this.state.registerOtpVerified = true;
      this.updateRegisterButtonState();
      this.updateRegisterOtpCountdownText('OTP verified successfully. You can now click Register.', 'active');
      if (!silent) {
        this.toast('OTP Verified', 'OTP is correct. Register button is now available.', 'ok');
      }
    } catch (error) {
      this.state.registerOtpVerified = false;
      this.updateRegisterButtonState();
      this.setFieldError(this.el.registerOtp);
      this.updateRegisterOtpCountdownText('Incorrect OTP. Please try again.', 'expired');
      if (!silent) {
        this.toast('OTP Verification Failed', this.parseApiError(error), 'error');
      }
    }
  }

  wireWorkspaceEvents() {
    this.setupTabs();
    this.setupExcelImport();
    this.wireEvents();
    this.el.adminRefresh?.addEventListener('click', () => this.loadAdminUsers());
    this.el.adminFeedbackRefresh?.addEventListener('click', () => this.loadAdminFeedback());

    // Manifest drawer toggle
    const drawer = document.getElementById('manifest-drawer');
    const toggleBtn = document.getElementById('btn-toggle-drawer');
    const closeBtn = document.getElementById('btn-close-drawer');
    if (drawer && toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        drawer.classList.toggle('open');
      });
    }
    if (drawer && closeBtn) {
      closeBtn.addEventListener('click', () => {
        drawer.classList.remove('open');
      });
    }

    // View-switching and Sub-Controllers init
    this.setupViewSwitching();
    this.fleetController.init();
    this.settingsController.init();
  }

  setupViewSwitching() {
    const navItems = document.querySelectorAll('.saas-nav .nav-item');
    navItems.forEach((item) => {
      const viewName = item.getAttribute('data-view');
      if (!viewName) return;
      item.addEventListener('click', (e) => {
        e.preventDefault();
        this.switchView(viewName);
      });
    });
    this.state.currentView = 'dispatch';
  }

  switchView(viewName) {
    this.state.currentView = viewName;

    // 1. Navigation items state
    const navItems = document.querySelectorAll('.saas-nav .nav-item');
    navItems.forEach((item) => {
      const itemView = item.getAttribute('data-view');
      item.classList.toggle('active', itemView === viewName);
    });

    // 2. Toggle panels
    const viewPanels = document.querySelectorAll('.view-panel');
    viewPanels.forEach((panel) => {
      const panelId = panel.id;
      const isTarget = panelId === `view-${viewName}`;
      panel.classList.toggle('hidden', !isTarget);
    });

    // 3. Header visual adjustments
    const pageTitleEl = document.querySelector('.saas-header .page-title');
    const headerRightEl = document.querySelector('.saas-header .header-right');

    if (viewName === 'dispatch') {
      if (pageTitleEl) pageTitleEl.textContent = this.lang === 'vn' ? 'Tối ưu hóa Tuyến đường' : 'Route Optimization';
      if (headerRightEl) headerRightEl.classList.remove('hidden');
      if (this.mapController) {
        setTimeout(() => this.mapController.invalidate(), 100);
      }
    } else {
      if (headerRightEl) headerRightEl.classList.add('hidden');
      if (viewName === 'fleet') {
        if (pageTitleEl) pageTitleEl.textContent = this.lang === 'vn' ? 'Cấu hình Đội xe' : 'Fleet Configuration';
        this.fleetController.render();
      } else if (viewName === 'analytics') {
        if (pageTitleEl)
          pageTitleEl.textContent = this.lang === 'vn' ? 'Phân tích Mô hình & AI' : 'Model Analytics & AI Diagnostics';
        if (!this.state.analysisData) {
          this.bootstrapAnalysis(true);
        }
      } else if (viewName === 'settings') {
        if (pageTitleEl) pageTitleEl.textContent = this.lang === 'vn' ? 'Cài đặt Hệ thống' : 'System Settings';
        this.settingsController.render();
      }
    }
  }

  enterApp() {
    this.state.unlocked = true;
    const isAuthPage = window.location.pathname.includes('auth.html') || window.location.pathname.endsWith('/auth');
    if (isAuthPage) {
      window.location.replace('app.html');
      return;
    }
    this.el.authScreen?.classList.add('hidden');
    this.el.appShell?.classList.remove('hidden');

    if (!this.mapController.ddqnMap) {
      this.mapController.init();
      this.simulationController.init();
      this.ganttController.init();
      this.wireWorkspaceEvents();
      this.renderCustomers();
      this.mapController.renderMarkers();
      this.showEmptyStates();
    } else {
      this.mapController.invalidate();
    }

    this.setStatus('Ready for operations.', 'ok');
    this.setImportEnabled(this.state.mode === 'real');

    // Start Firestore real-time telemetry only after the user is authenticated.
    // Calling onSnapshot before login triggers a known CORS bug on the emulator's
    // gRPC-Web streaming channel (/Listen/channel?TYPE=xmlhttp).
    if (db) {
      try {
        this.telemetry.startStreaming();
      } catch (e) {
        console.warn('[Telemetry] Firestore streaming unavailable:', e?.message || e);
      }
    }
    this.updateDatasetPickerVisibility();
    if (this.state.mode === 'sample') {
      this.loadAvailableDatasets().then(async () => {
        await this.loadSolomonDataset('demo');
        // Auto-execute optimization in demo mode for an immersive onboarding experience
        if (this.state.customers.length > 1 && !this.state.lastResult) {
          this.setStatus('Auto-optimizing demo fleet...', 'ok');
          this.toast('Demo Pipeline', 'Running full optimization pipeline automatically.', 'ok');
          await new Promise((r) => setTimeout(r, 800));
          this.submitJob();
        }
      });
    }
    this.bootstrapAnalysis();
    this.updateConnectionPill();
    this.updateSessionInfo();
    this.updateAdminPanel();
    this.activateTab(this.state.activeTab, true);
  }

  leaveApp() {
    this.state.unlocked = false;
    const isAppPage = window.location.pathname.includes('app.html') || window.location.pathname.endsWith('/app');
    if (isAppPage) {
      window.location.replace('auth.html');
      return;
    }
    this.el.appShell?.classList.add('hidden');
    this.el.authScreen?.classList.remove('hidden');
    if (this.el.authHint) {
      this.el.authHint.textContent =
        this.lang === 'vn'
          ? 'Đăng ký một lần rồi đăng nhập để nhận token.'
          : 'Register once, then log in to receive your token.';
    }
    this.showAuthView('login');
  }

  async initFirebase(email) {
    try {
      const enabled = await firebaseService.init(email);
      if (enabled) {
        await firebaseService.logEvent('login', { source: 'dashboard' });
        this.toast('Firebase Connected', 'Session data persistence is enabled.', 'ok');
      } else {
        this.toast('Firebase Not Configured', 'Fill in js/firebaseConfig.js to enable data persistence.', 'error');
      }
    } catch (error) {
      this.toast('Firebase Error', error.message, 'error');
    }
  }

  setupTabs() {
    this.el.tabButtons.forEach((button) => {
      button.addEventListener('click', () => this.activateTab(button.dataset.tab));
    });
    window.addEventListener('resize', () => this.updateTabIndicator());
    this.activateTab(this.state.activeTab, true);
  }

  setupExcelImport() {
    if (this.el.pickExcel) {
      this.el.pickExcel.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (this.state.mode !== 'real') return;
        if (this.el.excelInput) this.el.excelInput.value = '';
        this.el.excelInput?.click();
      });
    }

    if (this.el.excelInput) {
      this.el.excelInput.addEventListener('change', (event) => {
        if (this.state.mode !== 'real') {
          this.el.excelInput.value = '';
          return;
        }
        this.handleExcelFile(event);
      });
    }

    if (this.el.dropzone) {
      this.el.dropzone.addEventListener('click', (event) => {
        if (this.state.mode !== 'real') return;
        if (event.target === this.el.pickExcel) return;
        if (this.el.excelInput) this.el.excelInput.value = '';
        this.el.excelInput?.click();
      });
      this.el.dropzone.addEventListener('dragover', (event) => {
        if (this.state.mode !== 'real') return;
        event.preventDefault();
        this.el.dropzone.classList.add('dragover');
      });
      this.el.dropzone.addEventListener('dragleave', () => {
        this.el.dropzone.classList.remove('dragover');
      });
      this.el.dropzone.addEventListener('drop', (event) => {
        if (this.state.mode !== 'real') return;
        event.preventDefault();
        this.el.dropzone.classList.remove('dragover');
        const [file] = event.dataTransfer?.files ?? [];
        if (file) this.handleExcelFile({ target: { files: [file] } });
      });
    }
  }

  setImportEnabled(enabled) {
    const canImport = Boolean(enabled);
    if (this.el.pickExcel) this.el.pickExcel.disabled = !canImport;
    if (this.el.excelInput) this.el.excelInput.disabled = !canImport;
    if (this.el.dropzone) {
      this.el.dropzone.classList.toggle('disabled', !canImport);
      this.el.dropzone.setAttribute('aria-disabled', canImport ? 'false' : 'true');
      if (!canImport) this.el.dropzone.classList.remove('dragover');
    }
  }

  clearCustomersForRealDataMode() {
    this.state.customers = [];
    this.selectedCustomerIds.clear();
    this.resetResultOutputs();
    this.renderCustomers();
    this.mapController.renderMarkers();
  }

  wireEvents() {
    this.setupDirectPasteImport();

    this.el.modeToggle.addEventListener('change', () => {
      this.state.mode = this.el.modeToggle.checked ? 'real' : 'sample';
      this.updateDatasetPickerVisibility();
      if (this.state.mode === 'sample') {
        this.setImportEnabled(false);
        this.loadAvailableDatasets().then(() => {
          const selected = this.el.datasetSelect?.value || 'demo';
          this.loadSolomonDataset(selected);
        });
      } else {
        this.setImportEnabled(true);
        this.clearCustomersForRealDataMode();
        this.setStatus('Switched to Real Data mode. Use Excel/Pin input for your own dataset.', 'ok');
      }
    });

    this.el.loadDataset?.addEventListener('click', () => {
      const name = this.el.datasetSelect?.value || 'demo';
      this.loadSolomonDataset(name);
    });

    this.el.datasetSelect?.addEventListener('change', () => {
      const name = this.el.datasetSelect.value || 'demo';
      this.loadSolomonDataset(name);
    });

    this.el.vehicles.addEventListener('input', () => {
      this.state.vehicles = Number(this.el.vehicles.value);
      this.el.vehiclesValue.textContent = String(this.state.vehicles);
    });

    this.el.capacity.addEventListener('input', () => {
      this.state.capacity = Number(this.el.capacity.value);
      this.el.capacityValue.textContent = String(this.state.capacity);
    });

    this.el.addRow?.addEventListener('click', () => {
      this.tableInputVisible = true;
      this.renderCustomers();
      window.requestAnimationFrame(() => this.focusTableInputField('name'));
    });

    this.el.deleteSelected?.addEventListener('click', () => this.deleteSelectedCustomers());

    this.el.parsePaste?.addEventListener('click', () => this.parsePasteData());
    this.el.runModel.addEventListener('click', () => this.submitJob());
    this.el.addAddress?.addEventListener('click', () => this.addSelectedAddress());
    this.el.addressInput?.addEventListener('input', () => this.handleAddressInput());
    this.el.analysisVersion?.addEventListener('change', () => {
      const nextVersion = this.el.analysisVersion.value;
      if (!nextVersion) return;
      this.state.analysisVersion = nextVersion;
      this.loadAnalysisData(nextVersion);
    });
    this.el.analysisInstance?.addEventListener('change', () => {
      this.state.analysisInstance = this.el.analysisInstance.value || 'ALL';
      this.renderAnalysis();
    });
    this.el.analysisOpenPopup?.addEventListener('click', () => this.openAnalysisModal());
    this.el.analysisModalClose?.addEventListener('click', () => this.closeAnalysisModal());
    this.el.analysisModal?.addEventListener('click', (event) => {
      if (event.target === this.el.analysisModal) this.closeAnalysisModal();
    });
    window.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') this.closeAnalysisModal();
    });
    this.wireTableInlineEditing();
  }

  setupDirectPasteImport() {
    document.addEventListener('paste', async (event) => {
      if (!this.state.unlocked || this.state.mode !== 'real') return;

      const target = event.target;
      if (target instanceof HTMLElement) {
        const tag = target.tagName.toLowerCase();
        if (target.isContentEditable || tag === 'input' || tag === 'textarea' || tag === 'select') {
          return;
        }
      }

      const text = event.clipboardData?.getData('text/plain')?.trim() || '';
      if (!text) return;
      if (!text.includes('\n') && !text.includes('\t')) return;

      event.preventDefault();
      await this.importCustomersFromText(text, 'clipboard');
    });
  }

  async loadAvailableDatasets() {
    if (!this.el.datasetSelect) return;
    try {
      const data = await this.request('/solomon/list', { method: 'GET' });
      const datasets = Array.isArray(data?.datasets) ? data.datasets : [];
      if (datasets.length === 0) return;

      const currentValue = this.el.datasetSelect.value || 'demo';
      this.el.datasetSelect.replaceChildren(
        ...datasets.map((ds) => {
          const option = document.createElement('option');
          option.value = ds.name;
          option.textContent = ds.label || ds.name.toUpperCase();
          return option;
        })
      );
      const hasCurrentValue = datasets.some((ds) => ds.name === currentValue);
      this.el.datasetSelect.value = hasCurrentValue ? currentValue : 'demo';
    } catch (error) {

      this.toast('Solomon List Failed', this.parseApiError(error), 'error');
    }
  }

  updateDatasetPickerVisibility() {
    if (!this.el.datasetPicker) return;
    if (this.state.mode === 'sample') {
      this.el.datasetPicker.classList.remove('hidden');
    } else {
      this.el.datasetPicker.classList.add('hidden');
    }
  }

  async loadSolomonDataset(name = 'demo') {
    try {
      const data = await this.request(`/solomon?name=${encodeURIComponent(name)}`, { method: 'GET' });
      const incoming = Array.isArray(data?.customers) ? data.customers : [];
      if (incoming.length < 2) throw new Error('Solomon dataset is empty or invalid.');

      this.state.customers = incoming.map((c, idx) => ({
        ...c,
        id: idx,
        demand: Number(c.demand) || 0,
        ready: Number.isFinite(Number(c.ready)) ? Number(c.ready) : 0,
        due: Number.isFinite(Number(c.due)) ? Number(c.due) : 1000,
        service: Number.isFinite(Number(c.service)) ? Number(c.service) : 10,
        isDepot: Boolean(c.isDepot),
        priority: c.priority || (idx === 0 ? 'Normal' : ['Normal', 'High', 'Low'][idx % 3]),
        skill: c.skill || (idx === 0 ? 'None' : idx % 5 === 1 ? 'Refrigerated' : idx % 5 === 2 ? 'Hazmat' : 'None'),
      }));

      // Sync the dataset dropdown if available
      if (this.el.datasetSelect) {
        const options = Array.from(this.el.datasetSelect.options);
        if (options.some((opt) => opt.value === name)) {
          this.el.datasetSelect.value = name;
        }
      }

      // Dynamically adjust slider max to accommodate fleet config from the dataset
      const fleetVehicles = Math.max(1, Number(data?.fleet?.vehicles) || this.state.vehicles);
      const fleetCapacity = Math.max(1, Number(data?.fleet?.capacity) || this.state.capacity);

      if (this.el.vehicles && fleetVehicles > Number(this.el.vehicles.max)) {
        this.el.vehicles.max = String(Math.ceil(fleetVehicles * 1.5));
      }
      if (this.el.capacity && fleetCapacity > Number(this.el.capacity.max)) {
        this.el.capacity.max = String(Math.ceil(fleetCapacity * 1.2));
      }

      const maxVehicles = Math.max(1, Number(this.el.vehicles?.max) || 30);
      const maxCapacity = Math.max(1, Number(this.el.capacity?.max) || 1000);
      const nextVehicles = Math.min(maxVehicles, Math.max(1, fleetVehicles));
      const nextCapacity = Math.min(maxCapacity, Math.max(1, fleetCapacity));

      this.state.vehicles = nextVehicles;
      this.state.capacity = nextCapacity;
      if (this.el.vehicles) this.el.vehicles.value = String(nextVehicles);
      if (this.el.vehiclesValue) this.el.vehiclesValue.textContent = String(nextVehicles);
      if (this.el.capacity) this.el.capacity.value = String(nextCapacity);
      if (this.el.capacityValue) this.el.capacityValue.textContent = String(nextCapacity);

      this.selectedCustomerIds.clear();
      this.renderCustomers();
      this.mapController.renderMarkers();
      const label = String(data?.dataset || name).toUpperCase();
      this.setStatus(`Loaded Solomon ${label} with ${incoming.length - 1} customers.`, 'ok');
      this.toast('Solomon Loaded', `Dataset ${label} is ready.`, 'ok');
    } catch (error) {
      this.state.customers = [];
      this.selectedCustomerIds.clear();
      this.renderCustomers();
      this.mapController.renderMarkers();
      const message = this.parseApiError(error);
      this.toast('Solomon Load Failed', message, 'error');
      this.setStatus(message, 'error');
    }
  }

  async bootstrapAnalysis(forceReload = false) {
    if (!this.el.analysisVersion) return;
    if (!this.state.token) {
      this.clearAnalysisViews(
        this.lang === 'vn'
          ? 'Cần đăng nhập để tải phần phân tích huấn luyện.'
          : 'Login is required to load training analysis.'
      );
      return;
    }

    try {
      this.setAnalysisStatus(this.lang === 'vn' ? 'Đang tải các phiên bản...' : 'Loading available versions...');
      const response = await this.request('/analysis/versions', { method: 'GET' });
      const versions = Array.isArray(response?.items) ? response.items : [];
      this.state.analysisVersions = versions;
      this.renderAnalysisVersionOptions();

      let selectedVersion = this.state.analysisVersion;
      const hasSelected = versions.some((item) => item.version === selectedVersion);
      if (!selectedVersion || !hasSelected || forceReload) {
        selectedVersion = response?.default || versions[0]?.version || '';
      }

      if (!selectedVersion) {
        this.clearAnalysisViews(
          this.lang === 'vn'
            ? 'Không tìm thấy nexus_demo.json trong logs/results.'
            : 'No nexus_demo.json found in logs results folders.'
        );
        return;
      }

      this.state.analysisVersion = selectedVersion;
      this.el.analysisVersion.value = selectedVersion;
      await this.loadAnalysisData(selectedVersion);
      await this.loadAnalysisActivity();
    } catch (error) {
      this.clearAnalysisViews(this.parseApiError(error));
      this.toast('Analysis Load Failed', this.parseApiError(error), 'error');
    }
  }

  renderAnalysisVersionOptions() {
    if (!this.el.analysisVersion) return;
    this.el.analysisVersion.innerHTML = '';
    this.state.analysisVersions.forEach((item) => {
      const option = document.createElement('option');
      option.value = item.version;
      option.textContent = item.version.toUpperCase();
      this.el.analysisVersion.appendChild(option);
    });
  }

  async loadAnalysisData(version) {
    if (!version) return;
    try {
      const loadingLabel = this.lang === 'vn' ? 'Đang tải phân tích cho' : 'Loading analysis for';
      this.setAnalysisStatus(`${loadingLabel} ${String(version).toUpperCase()}...`);
      const payload = await this.request(`/analysis/nexus?version=${encodeURIComponent(version)}`, { method: 'GET' });
      this.state.analysisData = payload;
      this.renderAnalysis();
      const updated = this.state.analysisVersions.find((item) => item.version === version)?.updated_at;
      const stamp = updated ? new Date(updated).toLocaleString() : 'Unknown timestamp';
      if (this.el.analysisLastUpdated) {
        this.el.analysisLastUpdated.textContent = `Version ${String(version).toUpperCase()} • updated ${stamp}`;
      }
      this.setAnalysisStatus(
        this.lang === 'vn'
          ? 'Phân tích sẵn sàng. Mở Deep Analysis để xem đầy đủ.'
          : 'Analysis ready. Open Deep Analysis for full diagnostics.'
      );
    } catch (error) {
      this.clearAnalysisViews(this.parseApiError(error));
      this.toast('Analysis Version Failed', this.parseApiError(error), 'error');
    }
  }

  async loadAnalysisActivity() {
    try {
      const payload = await this.request('/analysis/activity?hours=24', { method: 'GET' });
      this.state.analysisActivity = payload;
      this.renderActivityChart(this.el.analysisActivityChart, payload);
    } catch (error) {
      this.state.analysisActivity = null;
      if (this.el.analysisActivityChart) {
        this.el.analysisActivityChart.textContent = this.parseApiError(error);
      }
    }
  }

  setAnalysisStatus(message) {
    if (!this.el.analysisStatus) return;
    this.el.analysisStatus.textContent = message;
    this.el.analysisStatus.className = 'status ok';
  }

  clearAnalysisViews(message) {
    if (this.el.analysisStatus) {
      this.el.analysisStatus.className = 'status error';
      this.el.analysisStatus.textContent = message;
    }
    if (this.el.analysisSummaryKpis) this.el.analysisSummaryKpis.innerHTML = '';
    if (this.el.analysisInstance) this.el.analysisInstance.innerHTML = '';
    if (this.el.analysisConvergenceChart)
      this.el.analysisConvergenceChart.textContent =
        this.lang === 'vn' ? 'Chưa có dữ liệu hội tụ.' : 'No convergence data yet.';
    if (this.el.analysisActivityChart)
      this.el.analysisActivityChart.textContent =
        this.state.analysisNoData || (this.lang === 'vn' ? 'Chưa có dữ liệu gantt job.' : 'No job timeline data yet.');
    if (this.el.analysisPolicyGrid)
      this.el.analysisPolicyGrid.textContent =
        this.lang === 'vn' ? 'Chưa có policy matrix.' : 'No policy matrix available.';
    if (this.el.analysisLeaderboardBody) this.el.analysisLeaderboardBody.innerHTML = '';
    if (this.el.analysisTransferBody) this.el.analysisTransferBody.innerHTML = '';
    if (this.el.analysisModalMeta)
      this.el.analysisModalMeta.textContent = this.lang === 'vn' ? 'Chưa có metadata.' : 'No metadata available.';
    if (this.el.analysisModalConvergence)
      this.el.analysisModalConvergence.textContent =
        this.lang === 'vn' ? 'Chưa có lịch sử hội tụ.' : 'No convergence history.';
    if (this.el.analysisModalTransferPlot)
      this.el.analysisModalTransferPlot.textContent =
        this.lang === 'vn' ? 'Chưa có dữ liệu transfer.' : 'No transfer data.';
    if (this.el.analysisModalTransferBody) this.el.analysisModalTransferBody.innerHTML = '';
  }

  renderAnalysis() {
    const data = this.state.analysisData;
    if (!data) {
      this.clearAnalysisViews('No analysis payload available.');
      return;
    }

    const summaryRows = Array.isArray(data.summary) ? data.summary : [];
    const transferRows = Array.isArray(data.transfer) ? data.transfer : [];
    const pairMap = this.buildSummaryPairMap(summaryRows);
    const instances = Array.from(pairMap.entries()).sort((a, b) => a[0].localeCompare(b[0]));
    const availableInstances = this.buildAvailableInstances(summaryRows, transferRows);
    const fallbackInstance = 'ALL';
    if (this.state.analysisInstance !== 'ALL' && !availableInstances.includes(this.state.analysisInstance)) {
      this.state.analysisInstance = fallbackInstance;
    }
    this.renderAnalysisInstanceOptions(availableInstances);

    const selectedInstance = this.state.analysisInstance || 'ALL';
    const filteredInstances =
      selectedInstance === 'ALL' ? instances : instances.filter(([name]) => name === selectedInstance);
    const preferred =
      selectedInstance === 'ALL'
        ? pairMap.get(String(data?.meta?.instance || '')) || filteredInstances[0]?.[1] || null
        : pairMap.get(selectedInstance) || null;

    this.renderAnalysisKpis(filteredInstances, preferred, selectedInstance);
    this.renderConvergence(
      this.el.analysisConvergenceChart,
      data?.alns?.history,
      data?.rl_alns?.history,
      selectedInstance,
      String(data?.meta?.instance || '')
    );
    this.renderPolicyHeatmap(data?.op_matrix, data?.destroy_ops, data?.repair_ops);
    this.renderLeaderboard(summaryRows, selectedInstance);
    this.renderTransferRows(transferRows, selectedInstance);

    if (!this.el.analysisModal?.classList.contains('hidden')) {
      this.renderAnalysisModal();
    }
  }

  buildAvailableInstances(summaryRows, transferRows) {
    const set = new Set();
    (Array.isArray(summaryRows) ? summaryRows : []).forEach((row) => {
      const value = String(row?.instance || '').trim();
      if (value) set.add(value);
    });
    (Array.isArray(transferRows) ? transferRows : []).forEach((row) => {
      const value = String(row?.instance || '').trim();
      if (value) set.add(value);
    });
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }

  renderAnalysisInstanceOptions(instances) {
    if (!this.el.analysisInstance) return;
    const current = this.state.analysisInstance || 'ALL';
    const options = ['ALL', ...instances];
    this.el.analysisInstance.replaceChildren(
      ...options.map((value) => {
        const option = document.createElement('option');
        option.value = String(value);
        option.textContent = value === 'ALL' ? 'ALL INSTANCES' : String(value);
        return option;
      })
    );
    this.el.analysisInstance.value = options.includes(current) ? current : 'ALL';
    this.state.analysisInstance = this.el.analysisInstance.value;
  }

  buildSummaryPairMap(rows) {
    const map = new Map();
    rows.forEach((row) => {
      const instance = String(row?.instance || '').trim();
      const algo = String(row?.algo || '')
        .trim()
        .toUpperCase();
      if (!instance || !algo) return;
      if (!map.has(instance)) map.set(instance, {});
      const entry = map.get(instance);
      if (algo === 'ALNS') entry.alns = row;
      if (algo === 'DDQN-ALNS') entry.ddqn = row;
    });
    return map;
  }

  renderAnalysisKpis(instances, preferredPair, preferredInstance) {
    if (!this.el.analysisSummaryKpis) return;

    const valid = instances.filter(([, pair]) => pair?.alns && pair?.ddqn);
    const total = valid.length;
    const gapWins = valid.filter(([, pair]) => Number(pair.ddqn.gap_pct) < Number(pair.alns.gap_pct)).length;
    const speedups = valid
      .map(([, pair]) => {
        const alnsTime = Number(pair.alns.time_s);
        const ddqnTime = Number(pair.ddqn.time_s);
        if (!Number.isFinite(alnsTime) || alnsTime <= 0 || !Number.isFinite(ddqnTime)) return null;
        return ((alnsTime - ddqnTime) / alnsTime) * 100;
      })
      .filter((value) => value !== null);
    const avgSpeedup = speedups.length ? speedups.reduce((sum, value) => sum + value, 0) / speedups.length : 0;
    const stabilityWins = valid.filter(([, pair]) => Number(pair.ddqn.td_cv) < Number(pair.alns.td_cv)).length;
    const preferredGapDelta = preferredPair
      ? Number(preferredPair.ddqn.gap_pct) - Number(preferredPair.alns.gap_pct)
      : 0;

    const cards = [
      {
        label: `Gap Delta (${preferredInstance || 'N/A'})`,
        value: `${preferredGapDelta >= 0 ? '+' : ''}${preferredGapDelta.toFixed(2)}%`,
        note: 'Negative means DDQN is closer to BKS',
      },
      {
        label: 'DDQN Gap Wins',
        value: `${gapWins}/${total || 0}`,
        note: 'Instances where DDQN gap < ALNS gap',
      },
      {
        label: 'Average Speedup',
        value: `${avgSpeedup.toFixed(1)}%`,
        note: 'Runtime reduction of DDQN vs ALNS',
      },
      {
        label: 'Stability Wins',
        value: `${stabilityWins}/${total || 0}`,
        note: 'Instances where DDQN TD_CV is lower',
      },
    ];

    this.el.analysisSummaryKpis.innerHTML = cards
      .map(
        (card) => `
          <article class="analysis-kpi">
            <span class="analysis-kpi-label">${card.label}</span>
            <strong class="analysis-kpi-value">${card.value}</strong>
            <span class="analysis-kpi-note">${card.note}</span>
          </article>
        `
      )
      .join('');
  }

  renderConvergence(container, alnsHistory, ddqnHistory, selectedInstance = 'ALL', historyInstance = '') {
    if (!container) return;
    const a = Array.isArray(alnsHistory)
      ? alnsHistory.map((value) => Number(value)).filter((value) => Number.isFinite(value))
      : [];
    const b = Array.isArray(ddqnHistory)
      ? ddqnHistory.map((value) => Number(value)).filter((value) => Number.isFinite(value))
      : [];
    if (a.length < 2 && b.length < 2) {
      container.textContent = 'No convergence data yet.';
      return;
    }

    const allValues = [...a, ...b];
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const span = Math.max(1e-9, max - min);
    const width = 700;
    const height = 180;
    const paddingX = 24;
    const paddingY = 12;

    const buildPath = (values) => {
      if (values.length === 0) return '';
      return values
        .map((value, index) => {
          const x = paddingX + (index / Math.max(1, values.length - 1)) * (width - paddingX * 2);
          const y = height - paddingY - ((value - min) / span) * (height - paddingY * 2);
          return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)} ${y.toFixed(2)}`;
        })
        .join(' ');
    };

    const pathA = buildPath(a);
    const pathB = buildPath(b);
    const showHistoryHint = selectedInstance !== 'ALL' && historyInstance && selectedInstance !== historyInstance;
    container.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Convergence chart">
        <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
        <line x1="${paddingX}" y1="${height - paddingY}" x2="${width - paddingX}" y2="${height - paddingY}" stroke="#d9e5f1" stroke-width="1" />
        <line x1="${paddingX}" y1="${paddingY}" x2="${paddingX}" y2="${height - paddingY}" stroke="#d9e5f1" stroke-width="1" />
        <path d="${pathA}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />
        <path d="${pathB}" fill="none" stroke="#0b8a65" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />
      </svg>
      <div class="analysis-legend">
        <span><i class="legend-dot" style="background:#2563eb"></i> ALNS</span>
        <span><i class="legend-dot" style="background:#0b8a65"></i> DDQN-ALNS</span>
        ${showHistoryHint ? `<span>History available for ${this.escapeHtml(String(historyInstance))}</span>` : ''}
      </div>
    `;
  }

  renderActivityChart(container, data) {
    if (!container) return;
    const labels = Array.isArray(data?.labels) ? data.labels : [];
    const submitted = Array.isArray(data?.submitted) ? data.submitted.map((value) => Number(value) || 0) : [];
    const completed = Array.isArray(data?.completed) ? data.completed.map((value) => Number(value) || 0) : [];
    const failed = Array.isArray(data?.failed) ? data.failed.map((value) => Number(value) || 0) : [];

    if (!labels.length || (!submitted.length && !completed.length && !failed.length)) {
      container.textContent =
        this.state.analysisNoData ||
        (this.lang === 'vn' ? 'Chưa có dữ liệu hoạt động theo giờ.' : 'No activity data yet.');
      return;
    }

    const maxCount = Math.max(1, ...submitted, ...completed, ...failed);
    const width = 740;
    const height = 240;
    const paddingX = 28;
    const paddingY = 24;
    const bandWidth = (width - paddingX * 2) / Math.max(labels.length, 1);
    const barWidth = Math.min(14, bandWidth / 4);

    const bars = labels
      .map((label, index) => {
        const baseX = paddingX + index * bandWidth + bandWidth / 2;
        const submittedHeight = ((submitted[index] || 0) / maxCount) * (height - paddingY * 2 - 24);
        const completedHeight = ((completed[index] || 0) / maxCount) * (height - paddingY * 2 - 24);
        const failedHeight = ((failed[index] || 0) / maxCount) * (height - paddingY * 2 - 24);
        const yBase = height - paddingY - 18;
        return `
        <g>
          <rect x="${(baseX - barWidth * 1.8).toFixed(2)}" y="${(yBase - submittedHeight).toFixed(2)}" width="${barWidth.toFixed(2)}" height="${Math.max(submittedHeight, 1).toFixed(2)}" rx="3" fill="#2563eb" />
          <rect x="${baseX.toFixed(2)}" y="${(yBase - completedHeight).toFixed(2)}" width="${barWidth.toFixed(2)}" height="${Math.max(completedHeight, 1).toFixed(2)}" rx="3" fill="#0b8a65" />
          <rect x="${(baseX + barWidth * 1.8).toFixed(2)}" y="${(yBase - failedHeight).toFixed(2)}" width="${barWidth.toFixed(2)}" height="${Math.max(failedHeight, 1).toFixed(2)}" rx="3" fill="#c0392b" />
          <text x="${baseX.toFixed(2)}" y="${(height - 8).toFixed(2)}" text-anchor="middle" font-size="10" fill="#54708a">${this.escapeHtml(String(label))}</text>
        </g>
      `;
      })
      .join('');

    const totalSubmitted = submitted.reduce((sum, value) => sum + value, 0);
    const totalCompleted = completed.reduce((sum, value) => sum + value, 0);
    const totalFailed = failed.reduce((sum, value) => sum + value, 0);
    const rangeLabel = `${String(data?.hours || labels.length)}h ${this.lang === 'vn' ? 'gần nhất' : 'latest'}`;
    const seriesText =
      this.lang === 'vn'
        ? { submitted: 'Đã gửi', completed: 'Hoàn thành', failed: 'Lỗi' }
        : { submitted: 'Submitted', completed: 'Completed', failed: 'Failed' };

    container.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Hourly activity chart">
        <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
        <line x1="${paddingX}" y1="${height - paddingY - 18}" x2="${width - paddingX}" y2="${height - paddingY - 18}" stroke="#d9e5f1" stroke-width="1" />
        <line x1="${paddingX}" y1="${paddingY}" x2="${paddingX}" y2="${height - paddingY - 18}" stroke="#d9e5f1" stroke-width="1" />
        ${bars}
      </svg>
      <div class="analysis-legend">
        <span><i class="legend-dot" style="background:#2563eb"></i> ${seriesText.submitted} ${totalSubmitted}</span>
        <span><i class="legend-dot" style="background:#0b8a65"></i> ${seriesText.completed} ${totalCompleted}</span>
        <span><i class="legend-dot" style="background:#c0392b"></i> ${seriesText.failed} ${totalFailed}</span>
        <span>${rangeLabel}</span>
      </div>
    `;
  }

  renderPolicyHeatmap(matrix, destroyOps, repairOps) {
    if (!this.el.analysisPolicyGrid) return;
    if (!Array.isArray(matrix) || matrix.length === 0) {
      this.el.analysisPolicyGrid.textContent = 'No policy matrix available.';
      return;
    }

    const rows = matrix.map((row) => (Array.isArray(row) ? row : []));
    const maxValue = Math.max(1, ...rows.flat().map((value) => Number(value) || 0));
    const colCount = Math.max(...rows.map((row) => row.length), 0);
    const columns = Array.from({ length: colCount }, (_, index) => String(repairOps?.[index] || `R${index + 1}`));

    const header = columns.map((name) => `<th>${this.escapeHtml(name)}</th>`).join('');
    const body = rows
      .map((row, rowIdx) => {
        const label = String(destroyOps?.[rowIdx] || `D${rowIdx + 1}`);
        const cells = columns
          .map((_, colIdx) => {
            const value = Number(row[colIdx]) || 0;
            const alpha = Math.max(0.08, value / maxValue);
            return `<td style="background: rgba(11,138,101,${alpha.toFixed(3)})">${value}</td>`;
          })
          .join('');
        return `<tr><th>${this.escapeHtml(label)}</th>${cells}</tr>`;
      })
      .join('');

    this.el.analysisPolicyGrid.innerHTML = `
      <table class="analysis-heatmap">
        <thead>
          <tr><th></th>${header}</tr>
        </thead>
        <tbody>${body}</tbody>
      </table>
    `;
  }

  renderLeaderboard(summaryRows, selectedInstance = 'ALL') {
    if (!this.el.analysisLeaderboardBody) return;
    this.el.analysisLeaderboardBody.innerHTML = '';

    const filtered =
      selectedInstance === 'ALL'
        ? summaryRows
        : summaryRows.filter((r) => String(r?.instance || '') === selectedInstance);

    // Group by instance, then sort each group by gap_pct ascending
    const byInstance = new Map();
    filtered.forEach((row) => {
      const inst = String(row?.instance || '').trim();
      if (!inst) return;
      if (!byInstance.has(inst)) byInstance.set(inst, []);
      byInstance.get(inst).push(row);
    });

    const ALGO_COLORS = {
      'DDQN-ALNS': '#059669',
      ALNS: '#2563eb',
      'OR-TOOLS': '#7c3aed',
      'HYBRID-FIXED': '#d97706',
      'HYBRID-RULE': '#ea6c00',
      'HYBRID-DDQN-TRANSFER-DR': '#0891b2',
    };
    const MEDALS = ['🥇', '🥈', '🥉'];

    const sortedInstances = Array.from(byInstance.keys()).sort((a, b) => a.localeCompare(b));
    sortedInstances.forEach((inst) => {
      const algos = byInstance.get(inst).sort((a, b) => Number(a.gap_pct) - Number(b.gap_pct));
      algos.forEach((row, rank) => {
        const tr = document.createElement('tr');
        const algoName = String(row.algo || '').toUpperCase();
        const color = ALGO_COLORS[algoName] || '#6b7280';
        const medal = rank < 3 ? MEDALS[rank] : `#${rank + 1}`;
        const gap = Number(row.gap_pct);
        const gapClass = gap <= 0 ? 'color: #059669; font-weight: 600' : gap < 3 ? 'color: #d97706' : 'color: #dc2626';
        const stability = Number(row.td_cv);
        const stabText = Number.isFinite(stability) ? stability.toFixed(2) + '%' : '—';

        tr.innerHTML = `
          <td>${rank === 0 ? this.escapeHtml(inst) : '<span style="color:#aaa">↳</span>'}</td>
          <td><span class="algo-tag" style="background:${color}15; color:${color}; border:1px solid ${color}40; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600">${this.escapeHtml(String(row.algo || ''))}</span></td>
          <td class="num" style="font-family:var(--font-data)">${Number(row.td || 0).toFixed(2)}</td>
          <td class="num" style="font-family:var(--font-data)">${Number(row.nv || 0).toFixed(1)}</td>
          <td class="num" style="font-family:var(--font-data); ${gapClass}">${gap >= 0 ? '+' : ''}${gap.toFixed(2)}%</td>
          <td class="num" style="font-family:var(--font-data)">${Number(row.time_s || 0).toFixed(1)}s</td>
          <td class="num" style="font-family:var(--font-data)">${stabText}</td>
          <td style="text-align:center; font-size:14px">${medal}</td>
        `;
        if (rank === 0) tr.style.background = `${color}08`;
        this.el.analysisLeaderboardBody.appendChild(tr);
      });
    });
  }

  renderTransferRows(rows, selectedInstance = 'ALL') {
    if (!this.el.analysisTransferBody) return;
    this.el.analysisTransferBody.innerHTML = '';

    const filtered = rows.filter(
      (row) => selectedInstance === 'ALL' || String(row?.instance || '') === selectedInstance
    );

    filtered.forEach((row) => {
      const gap = Number(row.gap_pct || 0);
      const gapClass = gap <= 0 ? 'color: #059669; font-weight: 600' : gap < 3 ? 'color: #d97706' : 'color: #dc2626';
      const inflated = row.nv_inflated;
      const statusHtml = inflated
        ? '<span class="analysis-pill bad" style="font-size:10px">Over-fleet</span>'
        : '<span class="analysis-pill good" style="font-size:10px">Optimal</span>';

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${this.escapeHtml(String(row.instance || '-'))}</td>
        <td>${this.escapeHtml(String(row.dataset || '-'))}</td>
        <td><span style="font-weight:600; font-size:11px">${this.escapeHtml(String(row.algo || '-'))}</span></td>
        <td class="num" style="font-family:var(--font-data)">${Number(row.td || 0).toFixed(2)}</td>
        <td class="num" style="font-family:var(--font-data)">${Number(row.nv || 0).toFixed(1)}</td>
        <td class="num" style="font-family:var(--font-data); ${gapClass}">${gap >= 0 ? '+' : ''}${gap.toFixed(2)}%</td>
        <td class="num" style="font-family:var(--font-data)">${Number(row.time_s || 0).toFixed(1)}s</td>
        <td>${statusHtml}</td>
      `;
      this.el.analysisTransferBody.appendChild(tr);
    });
  }

  renderPill(label) {
    if (label === 'DDQN') return '<span class="analysis-pill good">DDQN</span>';
    if (label === 'ALNS') return '<span class="analysis-pill bad">ALNS</span>';
    return '<span class="analysis-pill neutral">Tie</span>';
  }

  openAnalysisModal() {
    if (!this.state.analysisData) {
      this.toast('No Analysis Yet', 'Please load an analysis version first.', 'error');
      return;
    }
    this.renderAnalysisModal();
    this.el.analysisModal?.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
  }

  closeAnalysisModal() {
    if (this.el.analysisModal?.classList.contains('hidden')) return;
    this.el.analysisModal.classList.add('hidden');
    document.body.style.overflow = '';
  }

  renderAnalysisModal() {
    const data = this.state.analysisData;
    if (!data) return;

    const meta = data.meta || {};
    const source = data._source || {};
    const selectedInstance = this.state.analysisInstance || 'ALL';
    if (this.el.analysisModalSubtitle) {
      const filterLabel = selectedInstance === 'ALL' ? 'ALL INSTANCES' : selectedInstance;
      this.el.analysisModalSubtitle.textContent = `Version ${String(source.version || this.state.analysisVersion || '').toUpperCase()} • ${String(meta.dataset || 'Unknown dataset')} • Filter ${filterLabel}`;
    }

    if (this.el.analysisModalMeta) {
      const entries = [
        ['Instance', meta.instance],
        ['Customers', meta.n_customers],
        ['Capacity', meta.capacity],
        ['Horizon', meta.horizon],
        ['Dataset', meta.dataset],
        ['Version', meta.version || source.version],
      ];
      this.el.analysisModalMeta.innerHTML = entries
        .map(
          ([label, value]) =>
            `<div class="analysis-meta-item"><strong>${label}:</strong> ${this.escapeHtml(String(value ?? '-'))}</div>`
        )
        .join('');
    }

    this.renderConvergence(
      this.el.analysisModalConvergence,
      data?.alns?.history,
      data?.rl_alns?.history,
      selectedInstance,
      String(meta?.instance || '')
    );
    this.renderTransferPlot(data?.transfer, data?.summary, selectedInstance);
    this.renderModalTransferTable(data?.transfer, selectedInstance);
  }

  renderTransferPlot(transferRows, summaryRows, selectedInstance = 'ALL') {
    if (!this.el.analysisModalTransferPlot) return;
    const transfer = Array.isArray(transferRows) ? transferRows : [];
    if (transfer.length === 0) {
      this.el.analysisModalTransferPlot.textContent = 'No transfer data.';
      return;
    }

    const summary = Array.isArray(summaryRows) ? summaryRows : [];
    const alnsMap = new Map();
    summary.forEach((row) => {
      if (String(row?.algo || '').toUpperCase() !== 'ALNS') return;
      alnsMap.set(String(row.instance || ''), Number(row.gap_pct));
    });

    const points = transfer
      .map((row) => {
        const instance = String(row.instance || '');
        if (selectedInstance !== 'ALL' && instance !== selectedInstance) return null;
        const x = alnsMap.get(instance);
        const y = Number(row.gap_pct);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        return { instance, x, y };
      })
      .filter((item) => item !== null);

    if (points.length === 0) {
      this.el.analysisModalTransferPlot.textContent = 'Transfer points are missing baseline ALNS gap values.';
      return;
    }

    const minX = Math.min(...points.map((p) => p.x));
    const maxX = Math.max(...points.map((p) => p.x));
    const minY = Math.min(...points.map((p) => p.y));
    const maxY = Math.max(...points.map((p) => p.y));
    const xSpan = Math.max(1e-9, maxX - minX);
    const ySpan = Math.max(1e-9, maxY - minY);
    const width = 700;
    const height = 220;
    const pad = 30;

    const circles = points
      .map((point) => {
        const cx = pad + ((point.x - minX) / xSpan) * (width - pad * 2);
        const cy = height - pad - ((point.y - minY) / ySpan) * (height - pad * 2);
        const safeLabel = this.escapeHtml(String(point.instance));
        return `<g><circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="5.5" fill="#ea8a1d"/><text x="${(cx + 7).toFixed(2)}" y="${(cy - 7).toFixed(2)}" font-size="11" fill="#35516a">${safeLabel}</text></g>`;
      })
      .join('');

    this.el.analysisModalTransferPlot.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Transfer scatter">
        <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
        <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#d9e5f1" stroke-width="1"/>
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#d9e5f1" stroke-width="1"/>
        ${circles}
      </svg>
      <div class="analysis-legend">
        <span><i class="legend-dot" style="background:#ea8a1d"></i> X: ALNS Gap% (RC2)</span>
        <span><i class="legend-dot" style="background:#ea8a1d"></i> Y: DDQN-ALNS★ Gap% (RC2)</span>
      </div>
    `;
  }

  renderModalTransferTable(rows, selectedInstance = 'ALL') {
    if (!this.el.analysisModalTransferBody) return;
    this.el.analysisModalTransferBody.innerHTML = '';
    (Array.isArray(rows) ? rows : [])
      .filter((row) => selectedInstance === 'ALL' || String(row?.instance || '') === selectedInstance)
      .forEach((row) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
        <td>${this.escapeHtml(String(row.instance || '-'))}</td>
        <td>${Number(row.td || 0).toFixed(2)}</td>
        <td>${Number(row.gap_pct || 0).toFixed(2)}%</td>
        <td>${Number(row.nv || 0).toFixed(1)}</td>
      `;
        this.el.analysisModalTransferBody.appendChild(tr);
      });
  }

  escapeHtml(value) {
    return String(value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  toggleCustomerSelection(customerId) {
    if (!Number.isFinite(customerId)) return;
    if (this.selectedCustomerIds.has(customerId)) this.selectedCustomerIds.delete(customerId);
    else this.selectedCustomerIds.add(customerId);
    this.renderCustomers();
  }

  deleteSelectedCustomers() {
    if (this.selectedCustomerIds.size === 0) {
      this.toast('No Selection', 'Select one or more rows to delete.', 'error');
      return;
    }

    const before = this.state.customers.length;
    this.state.customers = this.state.customers.filter((item) => {
      if (item.isDepot) return true;
      return !this.selectedCustomerIds.has(item.id);
    });
    this.state.customers = this.state.customers.map((item, idx) => ({ ...item, id: idx }));
    this.selectedCustomerIds.clear();

    const removed = before - this.state.customers.length;
    if (removed <= 0) {
      this.toast('Delete Skipped', 'Depot row cannot be deleted.', 'error');
      return;
    }

    this.renderCustomers();
    this.mapController.renderMarkers();
    this.setStatus(`Deleted ${removed} selected customer row(s).`, 'ok');
    this.toast('Rows Deleted', `Removed ${removed} row(s).`, 'ok');
  }

  wireTableInlineEditing() {
    if (!this.el.customerRows) return;
    this.el.customerRows.addEventListener('dblclick', (event) => {
      const cell = event.target.closest('td[data-editable="true"]');
      if (!cell) return;
      this.startCustomerCellEdit(cell);
    });
  }

  startCustomerCellEdit(cell) {
    if (!cell || cell.classList.contains('is-editing')) return;

    const row = cell.closest('tr');
    const customerId = Number(row?.dataset.customerId);
    const field = cell.dataset.field;
    if (!Number.isFinite(customerId) || !field) return;

    const customer = this.state.customers.find((item) => item.id === customerId);
    if (!customer) return;

    const numericFields = ['demand', 'ready', 'due', 'service'];
    const isNumeric = numericFields.includes(field);
    const originalValue = isNumeric ? String(customer[field] ?? 0) : String(customer[field] ?? '');
    const input = document.createElement('input');
    input.className = 'table-edit-input';
    input.type = isNumeric ? 'number' : 'text';
    input.value = originalValue;
    if (isNumeric) {
      input.min = '0';
      input.step = field === 'demand' ? '1' : 'any';
    }

    cell.classList.add('is-editing');
    cell.textContent = '';
    cell.appendChild(input);
    input.focus();
    input.select();

    let done = false;
    const commit = async () => {
      if (done) return;
      done = true;
      await this.commitCustomerCellEdit({ customerId, field, rawValue: input.value, originalValue });
    };

    const cancel = () => {
      if (done) return;
      done = true;
      this.renderCustomers();
    };

    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        commit();
        return;
      }
      if (event.key === 'Escape') {
        event.preventDefault();
        cancel();
      }
    });

    input.addEventListener('blur', () => {
      commit();
    });
  }

  async commitCustomerCellEdit({ customerId, field, rawValue, originalValue }) {
    const customer = this.state.customers.find((item) => item.id === customerId);
    if (!customer) {
      this.renderCustomers();
      return;
    }

    const nextValue = String(rawValue ?? '').trim();
    if (nextValue === originalValue) {
      this.renderCustomers();
      return;
    }

    if (field === 'name') {
      if (!nextValue) {
        this.toast('Invalid Name', 'Name cannot be empty.', 'error');
        this.renderCustomers();
        return;
      }
      customer.name = nextValue;
      this.renderCustomers();
      this.mapController.renderMarkers();
      this.setStatus('Customer name updated.', 'ok');
      return;
    }

    if (field === 'demand') {
      const demand = Number(nextValue);
      if (!Number.isFinite(demand) || demand < 0) {
        this.toast('Invalid Demand', 'Demand must be a number >= 0.', 'error');
        this.renderCustomers();
        return;
      }
      customer.demand = Math.round(demand);
      this.renderCustomers();
      this.mapController.renderMarkers();
      this.setStatus('Customer demand updated.', 'ok');
      return;
    }

    if (field === 'address') {
      if (!nextValue) {
        this.toast('Invalid Address', 'Address cannot be empty.', 'error');
        this.renderCustomers();
        return;
      }

      customer.address = nextValue;
      const geocoded = await this.tryGeocodeFromText(nextValue);
      if (geocoded) {
        customer.lat = geocoded.lat;
        customer.lng = geocoded.lng;
        this.setStatus('Address updated. Coordinates were refreshed automatically.', 'ok');
      } else {
        this.setStatus('Address updated, but geocoding could not refresh coordinates.', 'error');
        this.toast('Geocode Warning', 'Could not find coordinates for this address.', 'error');
      }
      this.renderCustomers();
      this.mapController.renderMarkers();
      return;
    }

    if (field === 'ready' || field === 'due' || field === 'service') {
      const n = Number(nextValue);
      if (!Number.isFinite(n) || n < 0) {
        this.toast('Invalid Value', `${field} must be a non-negative number.`, 'error');
        this.renderCustomers();
        return;
      }
      if (field === 'ready') {
        const due = Number(customer.due);
        if (Number.isFinite(due) && n >= due) {
          this.toast('Invalid Time Window', 'Ready must be < Due.', 'error');
          this.renderCustomers();
          return;
        }
      }
      if (field === 'due') {
        const ready = Number(customer.ready);
        if (Number.isFinite(ready) && n <= ready) {
          this.toast('Invalid Time Window', 'Due must be > Ready.', 'error');
          this.renderCustomers();
          return;
        }
      }
      customer[field] = n;
      this.renderCustomers();
      this.setStatus(`Customer ${field} updated.`, 'ok');
      return;
    }

    if (field === 'priority') {
      const allowed = ['Low', 'Normal', 'High'];
      const formatted = nextValue.charAt(0).toUpperCase() + nextValue.slice(1).toLowerCase();
      if (!allowed.includes(formatted)) {
        this.toast('Invalid Priority', 'Priority must be Low, Normal, or High.', 'error');
        this.renderCustomers();
        return;
      }
      customer.priority = formatted;
      this.renderCustomers();
      this.setStatus('Customer priority updated.', 'ok');
      return;
    }

    if (field === 'skill') {
      const allowed = Object.keys(SKILL_CONFIG);
      const matched = allowed.find((a) => a.toLowerCase() === nextValue.toLowerCase()) || 'None';
      customer.skill = matched;
      this.renderCustomers();
      this.setStatus('Customer skill requirement updated.', 'ok');
      return;
    }

    this.renderCustomers();
  }

  activateTab(tabName, silent = false) {
    const availableTabs = new Set(this.el.tabButtons.map((button) => button.dataset.tab));
    const nextTab = availableTabs.has(tabName) ? tabName : 'overview';
    this.state.activeTab = nextTab;
    this.el.tabButtons.forEach((button) => {
      const isActive = button.dataset.tab === nextTab;
      button.classList.toggle('active', isActive);
      button.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
    let targetPanel = null;
    this.el.tabPanels.forEach((panel) => {
      const isActive = panel.dataset.panel === nextTab;
      panel.classList.toggle('active', isActive);
      panel.classList.remove('panel-focus');
      if (isActive && !targetPanel) targetPanel = panel;
    });

    if (targetPanel) {
      targetPanel.classList.add('panel-focus');
      if (!silent) {
        targetPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }

    this.updateTabIndicator();
    if (!silent) {
      this.toast('Tab Changed', `Switched to ${this.tabLabel(nextTab)}.`, 'ok');
    }
  }

  updateTabIndicator() {
    if (!this.el.tabbarIndicator) return;
    const activeButton = this.el.tabButtons.find((button) => button.classList.contains('active'));
    if (!activeButton) return;
    const parentRect = activeButton.parentElement?.getBoundingClientRect();
    const rect = activeButton.getBoundingClientRect();
    if (!parentRect) return;
    const offsetX = rect.left - parentRect.left;
    const offsetY = rect.top - parentRect.top;
    this.el.tabbarIndicator.style.width = `${rect.width}px`;
    this.el.tabbarIndicator.style.height = `${rect.height}px`;
    this.el.tabbarIndicator.style.transform = `translate3d(${offsetX}px, ${offsetY}px, 0)`;
  }

  tabLabel(tabName) {
    return { overview: 'Overview', maps: 'Split Map', results: 'Results', archive: 'Archive' }[tabName] ?? tabName;
  }

  async request(path, options = {}) {
    const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
    if (this.state.token) headers.Authorization = `Bearer ${this.state.token}`;
    try {
      const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
      if (!response.ok) {
        const contentType = response.headers.get('content-type') || '';
        if (contentType.includes('application/json')) {
          const data = await response.json().catch(() => null);
          throw new Error(data?.detail || data?.message || `HTTP ${response.status}`);
        }
        const body = await response.text();
        throw new Error(body || `HTTP ${response.status}`);
      }
      return response.json();
    } catch (error) {
      const message = String(error?.message || error || '');
      if (error instanceof TypeError || /failed to fetch|networkerror|load failed/i.test(message)) {
        throw new Error(`Cannot reach backend API at ${API_BASE}. Start the backend server on port 8000.`);
      }
      throw error;
    }
  }

  async register() {
    try {
      const email = this.el.registerEmail.value.trim().toLowerCase();
      const password = this.el.registerPassword.value.trim();
      const otp = this.el.registerOtp.value.trim();
      this.clearFieldError(this.el.registerEmail);
      this.clearFieldError(this.el.registerPassword);
      this.clearFieldError(this.el.registerOtp);

      if (!email || !this.isValidEmail(email)) {
        this.setFieldError(this.el.registerEmail);
        throw new Error('Invalid email format');
      }
      if (this.state.registerOtpApprovedEmail !== email) {
        this.setFieldError(this.el.registerEmail);
        throw new Error('Please send OTP successfully before registering');
      }
      if (!password) {
        this.setFieldError(this.el.registerPassword);
        throw new Error('Password is required');
      }
      if (!this.state.registerOtpVerified) {
        this.setFieldError(this.el.registerOtp);
        throw new Error('Please verify OTP before registering');
      }
      if (password.length < 6) {
        this.setFieldError(this.el.registerPassword);
        throw new Error('Password must be at least 6 characters');
      }
      if (!otp) {
        this.setFieldError(this.el.registerOtp);
        throw new Error('OTP is required');
      }
      if (!/^\d{6}$/.test(otp)) {
        this.setFieldError(this.el.registerOtp);
        throw new Error('OTP must be exactly 6 digits');
      }

      await this.request('/auth/register', {
        method: 'POST',
        body: JSON.stringify({ email, password, otp }),
      });
      this.toast('Registration Successful', 'Account created successfully.', 'ok');
      this.el.loginEmail.value = email;
      this.el.registerEmail.value = '';
      this.el.registerPassword.value = '';
      this.el.registerOtp.value = '';
      this.state.registerOtpApprovedEmail = '';
      this.registerOtpRequestedEmail = ''; // Clear on successful registration
      this.state.registerOtpVerified = false;
      this.state.registerOtpExpiresAt = 0;
      this.stopRegisterOtpCountdown();
      this.stopRegisterSuccessCountdown();
      this.updateRegisterOtpCountdownText('Click Send OTP to receive a verification code.');
      this.updateRegisterButtonState();
      this.showAuthView('login');
    } catch (error) {
      const message = this.parseApiError(error);
      if (/otp/i.test(message)) {
        this.setFieldError(this.el.registerOtp);
      } else if (/password/i.test(message)) {
        this.setFieldError(this.el.registerPassword);
      } else if (/email/i.test(message)) {
        this.setFieldError(this.el.registerEmail);
      }
      this.toast('Registration Failed', message, 'error');
    }
  }

  async probeBackendMode() {
    try {
      const response = await fetch(`${API_BASE}/health`, { method: 'GET' });
      if (!response.ok) return;
      const data = await response.json().catch(() => ({}));
      const safe = data && typeof data === 'object' ? data : {};
      const firebaseEnabled = 'firebase_enabled' in safe ? Boolean(safe.firebase_enabled) : true;
      const demoMode = 'demo_mode' in safe ? Boolean(safe.demo_mode) : !firebaseEnabled;
      this.backendMode.firebase_enabled = firebaseEnabled;
      this.backendMode.demo_mode = demoMode;
      this.backendMode.torch = safe && typeof safe.torch === 'object' ? safe.torch : null;
      this.applyBackendMode();
      this.applyDeviceBadge();
    } catch (error) {
      this.backendMode.firebase_enabled = null;
      this.backendMode.demo_mode = null;
      this.backendMode.torch = null;
      this.applyBackendMode();
      this.applyDeviceBadge();
    }
  }

  applyDeviceBadge() {
    const node = this.el.loadingDevice;
    if (!node) return;
    const torch = this.backendMode.torch;
    if (!torch || typeof torch !== 'object') {
      node.textContent = 'unknown';
      node.dataset.device = 'cpu';
      return;
    }
    if (torch.cuda_available) {
      const name = torch.device_name ? String(torch.device_name).replace(/^NVIDIA\s+/i, '') : 'GPU';
      node.textContent = `GPU - ${name}`;
      node.dataset.device = 'gpu';
    } else {
      node.textContent = 'CPU';
      node.dataset.device = 'cpu';
    }
  }

  applyBackendMode() {
    if (!this.el.guestBlock) return;
    const demo = this.backendMode.demo_mode;
    const fb = this.backendMode.firebase_enabled;
    const localAuthDisabled = this.isLocalAuthDisabled();
    // Show guest button when backend is in demo mode, or when we cannot probe
    const showGuest = demo === true || demo === null;
    this.el.guestBlock.classList.toggle('hidden', !showGuest);
    this.el.loginEmail && (this.el.loginEmail.disabled = localAuthDisabled);
    this.el.loginPassword && (this.el.loginPassword.disabled = localAuthDisabled);
    this.el.btnOpenRegister?.classList.toggle('hidden', localAuthDisabled);
    this.el.linkForgotPassword?.classList.toggle('hidden', localAuthDisabled);
    if (this.el.btnLogin && localAuthDisabled) {
      this.el.btnLogin.textContent = this.lang === 'vn' ? 'Vào Demo' : 'Continue Demo';
    } else if (this.el.btnLogin) {
      this.el.btnLogin.textContent = APP_COPY[this.lang]?.loginButton || 'Login';
    }
    if (this.el.authHint && localAuthDisabled && !this.state.unlocked) {
      this.el.authHint.textContent =
        this.lang === 'vn'
          ? 'Đăng nhập bằng email đang tắt vì Firebase chưa được cấu hình trên máy này. Dùng Demo để chạy solver.'
          : 'Email login is disabled because Firebase is not configured on this machine. Use Demo to run the solver.';
      this.el.authHint.style.display = 'block';
    }
    if (this.el.guestHint) {
      if (demo === true) {
        this.el.guestHint.textContent =
          'Local demo mode is active. Real email/password login needs Firebase credentials.';
      } else if (demo === null) {
        this.el.guestHint.textContent = 'Backend reachability unknown - guest mode kept available as fallback.';
      } else if (fb === true) {
        this.el.guestHint.textContent = 'Production auth is enabled. Guest access is disabled by the operator.';
      } else {
        this.el.guestHint.textContent = '';
      }
    }
  }

  isLocalAuthDisabled() {
    return this.backendMode.firebase_enabled === false && this.backendMode.demo_mode === true;
  }

  updateConnectionPill() {
    if (!this.el.connectionPill) return;
    const demo = this.backendMode.demo_mode;
    const hasToken = Boolean(this.state.token);
    this.el.connectionPill.className = 'pill soft';

    if (!hasToken) {
      this.el.connectionPill.textContent = 'Offline';
      return;
    }
    if (demo === true || this.state.role === 'guest') {
      this.el.connectionPill.textContent = 'Demo';
      this.el.connectionPill.classList.add('warn');
      return;
    }
    this.el.connectionPill.textContent = 'Online';
    this.el.connectionPill.classList.add('ok');
  }

  updateSessionInfo() {
    if (this.el.userEmail) {
      this.el.userEmail.textContent = this.state.email || '-';
    }
  }

  loginAsGuest() {
    this.state.token = 'demo-guest';
    this.state.email = 'guest@demo.local';
    this.state.role = 'guest';
    this.state.mustChangePassword = false;
    localStorage.setItem('vrptw_token', this.state.token);
    localStorage.setItem('vrptw_email', this.state.email);
    localStorage.setItem('vrptw_role', this.state.role);
    localStorage.setItem('vrptw_must_change_password', 'false');

    this.enterApp();
    this.updateConnectionPill();
    this.updateSessionInfo();
    this.toast('Demo Mode', 'Logged in as guest. Auth and admin endpoints stay disabled.', 'ok');
  }

  async login() {
    try {
      if (this.isLocalAuthDisabled()) {
        this.loginAsGuest();
        return;
      }

      const email = this.el.loginEmail.value.trim().toLowerCase();
      const password = this.el.loginPassword.value.trim();

      this.clearFieldError(this.el.loginEmail);
      this.clearFieldError(this.el.loginPassword);

      if (this.el.authHint) this.el.authHint.style.display = 'none';

      if (!this.isValidEmail(email)) {
        this.setFieldError(this.el.loginEmail);
        throw new Error('Invalid email format');
      }
      if (!password) {
        this.setFieldError(this.el.loginPassword);
        throw new Error('Please enter both email and password');
      }

      // NATIVE FIREBASE CALL via the service abstraction
      const firebaseUser = await firebaseService.loginUser(email, password);
      const idToken = await firebaseUser.getIdToken();

      this.state.token = idToken;
      this.state.email = firebaseUser.email;
      this.state.role = 'operator';

      // Security measure for Kiosks/TVMs: Wipe inputs from DOM memory
      this.el.loginEmail.value = '';
      this.el.loginPassword.value = '';

      localStorage.setItem('vrptw_token', this.state.token);
      localStorage.setItem('vrptw_email', this.state.email);
      localStorage.setItem('vrptw_role', this.state.role);
      localStorage.setItem('vrptw_must_change_password', 'false');

      this.enterApp();
      this.updateConnectionPill();
      this.updateSessionInfo();
      this.toast('Login Successful', 'Authenticated via Firebase.', 'ok');
    } catch (err) {
      console.error('[Login Failed]:', err);

      if (this.el.authHint) {
        let errorMsg = 'Invalid credentials or server error.';
        if (
          err.code === 'auth/invalid-credential' ||
          err.code === 'auth/user-not-found' ||
          err.code === 'auth/wrong-password'
        ) {
          errorMsg = 'Incorrect email or password. Please try again.';
        } else if (err.code === 'auth/too-many-requests') {
          errorMsg = 'Account temporarily locked due to many failed attempts.';
        } else if (err.message) {
          errorMsg = err.message;
        }

        this.el.authHint.textContent = errorMsg;
        this.el.authHint.style.display = 'block';
        this.el.authHint.style.color = '#e74c3c';
      }

      this.setFieldError(this.el.loginEmail);
      this.setFieldError(this.el.loginPassword);
    }
  }

  async requestForgotPasswordOtp() {
    if (this.isSendingForgotPasswordLink) return;

    try {
      this.isSendingForgotPasswordLink = true;
      this.el.btnForgotPassword && (this.el.btnForgotPassword.disabled = true);

      const email = this.el.forgotEmail.value.trim().toLowerCase();
      this.clearFieldError(this.el.forgotEmail);
      if (!this.isValidEmail(email)) {
        this.setFieldError(this.el.forgotEmail);
        throw new Error('Invalid email format');
      }

      const res = await this.request('/auth/forgot-password/request', {
        method: 'POST',
        body: JSON.stringify({ email }),
      });

      this.el.forgotEmail.value = '';
      this.el.loginEmail.value = email;
      this.toast(
        'Temporary Password Sent',
        `Delivery method: ${res.delivery}. Check your email for the temporary password.`,
        'ok'
      );
      this.showAuthView('login');
    } catch (error) {
      const message = this.parseApiError(error);
      this.setFieldError(this.el.forgotEmail);
      this.toast('Failed to Send Reset Link', message, 'error');
    } finally {
      this.isSendingForgotPasswordLink = false;
      this.el.btnForgotPassword && (this.el.btnForgotPassword.disabled = false);
    }
  }

  async resetForgotPassword() {
    if (this.isSubmittingPasswordChange) return;

    try {
      this.isSubmittingPasswordChange = true;
      this.el.btnResetPassword && (this.el.btnResetPassword.disabled = true);

      const password = this.el.resetPassword.value.trim();
      const confirm = this.el.resetPasswordConfirm.value.trim();
      this.clearFieldError(this.el.resetPassword);
      this.clearFieldError(this.el.resetPasswordConfirm);
      if (password.length < 6) {
        this.setFieldError(this.el.resetPassword);
        throw new Error('New password must be at least 6 characters');
      }
      if (password !== confirm) {
        this.setFieldError(this.el.resetPasswordConfirm);
        throw new Error('Password confirmation does not match');
      }

      if (this.state.mustChangePassword) {
        await this.request('/auth/password/change-required', {
          method: 'POST',
          body: JSON.stringify({ new_password: password }),
        });

        const loginEmail = this.state.email;
        this.state.mustChangePassword = false;
        this.state.resetToken = '';
        this.state.token = '';
        this.state.email = '';
        this.state.role = 'operator';
        localStorage.removeItem('vrptw_token');
        localStorage.removeItem('vrptw_email');
        localStorage.removeItem('vrptw_role');
        localStorage.removeItem('vrptw_must_change_password');
        this.el.resetPassword.value = '';
        this.el.resetPasswordConfirm.value = '';
        this.el.loginEmail.value = loginEmail;
        this.toast('Password Updated', 'New password saved. Please login again.', 'ok');
        this.showAuthView('login');
      } else {
        const token = this.state.resetToken || new URLSearchParams(window.location.search).get('token') || '';
        if (!token) throw new Error('Missing reset token in URL');

        await this.request('/auth/forgot-password/reset', {
          method: 'POST',
          body: JSON.stringify({ token, new_password: password }),
        });

        this.toast('Password Updated', 'You can now log in with your new password.', 'ok');
        this.state.resetToken = '';
        this.el.resetPassword.value = '';
        this.el.resetPasswordConfirm.value = '';
        window.history.replaceState({}, '', window.location.pathname);
        this.showAuthView('login');
      }
    } catch (error) {
      const message = this.parseApiError(error);
      this.toast('Failed to Update Password', message, 'error');
    } finally {
      this.isSubmittingPasswordChange = false;
      this.el.btnResetPassword && (this.el.btnResetPassword.disabled = false);
    }
  }

  async logout() {
    this.state.token = '';
    this.state.email = '';
    this.state.role = 'operator';
    this.state.mustChangePassword = false;
    localStorage.removeItem('vrptw_token');
    localStorage.removeItem('vrptw_email');
    localStorage.removeItem('vrptw_role');
    localStorage.removeItem('vrptw_must_change_password');
    this.updateConnectionPill();
    this.updateSessionInfo();
    this.leaveApp();
    this.toast('Logged Out', 'Session has ended.', 'ok');
    try {
      await firebaseService.logEvent('logout', { source: 'dashboard' });
    } catch {
      // Ignore logging failure on logout.
    }
  }

  updateAdminPanel() {
    const isAdmin = this.state.role === 'admin';
    this.el.adminPanel?.classList.toggle('hidden', !isAdmin);
    this.el.feedbackAdminLink?.classList.toggle('hidden', !isAdmin);
    if (isAdmin) {
      this.loadAdminUsers();
      this.loadAdminFeedback();
    }
  }

  async loadAdminUsers() {
    if (this.state.role !== 'admin' || !this.el.adminUserRows) return;
    try {
      const data = await this.request('/admin/users', { method: 'GET' });
      this.el.adminUserRows.replaceChildren();
      (data.items || []).forEach((item) => {
        const email = String(item.email || '');
        const role = ['admin', 'operator', 'viewer'].includes(item.role) ? item.role : 'operator';
        const createdAt = new Date(Number(item.created_at) * 1000).toLocaleString();

        const tr = document.createElement('tr');

        const emailCell = document.createElement('td');
        emailCell.textContent = email;
        tr.appendChild(emailCell);

        const roleCell = document.createElement('td');
        const select = document.createElement('select');
        select.className = 'admin-role-select';
        select.dataset.email = email;
        ['admin', 'operator', 'viewer'].forEach((value) => {
          const option = document.createElement('option');
          option.value = value;
          option.textContent = value;
          if (role === value) option.selected = true;
          select.appendChild(option);
        });
        roleCell.appendChild(select);
        tr.appendChild(roleCell);

        const createdCell = document.createElement('td');
        createdCell.textContent = createdAt;
        tr.appendChild(createdCell);

        const actionCell = document.createElement('td');
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn ghost';
        button.dataset.action = 'save-role';
        button.dataset.email = email;
        button.textContent = 'Save';
        actionCell.appendChild(button);
        tr.appendChild(actionCell);

        this.el.adminUserRows.appendChild(tr);
      });

      this.el.adminUserRows.querySelectorAll('[data-action="save-role"]').forEach((button) => {
        button.addEventListener('click', () => this.saveAdminRole(button.dataset.email));
      });
    } catch (error) {
      this.toast('Failed to Load User List', error.message, 'error');
    }
  }

  async loadAdminFeedback() {
    if (this.state.role !== 'admin' || !this.el.adminFeedbackRows) return;
    try {
      const data = await this.request('/admin/feedback', { method: 'GET' });
      const items = Array.isArray(data?.items) ? data.items : [];
      this.state.adminFeedback = items;
      this.renderAdminFeedback(items);
    } catch (error) {
      this.toast('Failed to Load Feedback', this.parseApiError(error), 'error');
    }
  }

  renderAdminFeedback(items) {
    if (!this.el.adminFeedbackRows) return;
    this.el.adminFeedbackRows.replaceChildren();
    (Array.isArray(items) ? items : []).forEach((item) => {
      const tr = document.createElement('tr');
      const when = item?.created_at ? new Date(Number(item.created_at) * 1000).toLocaleString() : '-';
      const rating = item?.rating ? String(item.rating) : '-';
      tr.innerHTML = `
        <td>${this.escapeHtml(String(when))}</td>
        <td>${this.escapeHtml(String(item?.category || '-'))}</td>
        <td>${this.escapeHtml(String(item?.message || '-'))}</td>
        <td>${this.escapeHtml(String(rating))}</td>
      `;
      this.el.adminFeedbackRows.appendChild(tr);
    });
  }

  async saveAdminRole(email) {
    if (!email || !this.el.adminUserRows) return;
    const select = Array.from(this.el.adminUserRows.querySelectorAll('select.admin-role-select')).find(
      (node) => node.dataset.email === email
    );
    if (!select) return;
    try {
      await this.request(`/admin/users/${encodeURIComponent(email)}/role`, {
        method: 'PATCH',
        body: JSON.stringify({ role: select.value }),
      });
      this.toast('Role Updated', `${email} -> ${select.value}`, 'ok');
    } catch (error) {
      this.toast('Failed to Update Role', error.message, 'error');
    }
  }

  async handleAddressInput() {
    const q = this.el.addressInput.value.trim();
    this.state.selectedSuggest = null;
    if (q.length < 3) {
      this.state.suggest = [];
      this.renderSuggest();
      return;
    }
    try {
      const data = await this.request(`/geocode?q=${encodeURIComponent(q)}&limit=6`, { method: 'GET' });
      this.state.suggest = data.items || [];
      this.renderSuggest();
    } catch {
      this.state.suggest = [];
      this.renderSuggest();
    }
  }

  renderSuggest() {
    this.el.suggestList.innerHTML = '';
    this.state.suggest.forEach((item) => {
      const li = document.createElement('li');
      li.textContent = item.address;
      li.addEventListener('click', () => {
        this.state.selectedSuggest = item;
        this.el.addressInput.value = item.address;
        this.state.suggest = [];
        this.renderSuggest();
      });
      this.el.suggestList.appendChild(li);
    });
  }

  addSelectedAddress() {
    const selected = this.state.selectedSuggest;
    if (!selected) {
      this.setStatus('Please select an address from the suggestion list.', 'error');
      return;
    }
    this.pushCustomer({
      name: `C-${Date.now().toString().slice(-4)}`,
      address: selected.address,
      lat: selected.lat,
      lng: selected.lng,
      demand: 10,
      ready: 0,
      due: 1000,
      service: 10,
    });
    this.el.addressInput.value = '';
    this.state.selectedSuggest = null;
    this.setStatus('Added a customer from address autocomplete.', 'ok');
    this.toast('Delivery Point Added', selected.address, 'ok');
  }

  async parsePasteData() {
    const text = this.el.pasteBox?.value?.trim() || '';
    if (!text) {
      this.setStatus('No data available to parse.', 'error');
      return;
    }

    await this.importCustomersFromText(text, 'clipboard');
    if (this.el.pasteBox) this.el.pasteBox.value = '';
  }

  parseTextRows(text) {
    const lines = String(text)
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    const useTabDelimiter = lines.some((line) => line.includes('\t'));
    const delimiter = useTabDelimiter ? /\t/ : /,/;
    return lines.map((line) => line.split(delimiter).map((cell) => String(cell).trim()));
  }

  async importCustomersFromText(text, source = 'clipboard') {
    const rows = this.parseTextRows(text);
    const newItems = await this.parseRowsToCustomers(rows);

    if (!newItems.length) {
      this.setStatus('No valid customer rows found in pasted data.', 'error');
      this.toast('Import Failed', 'No valid customer rows found in pasted data.', 'error');
      return;
    }

    newItems.forEach((item) => this.pushCustomer(item));
    this.setStatus(`Added ${newItems.length} customers from ${source} data.`, 'ok');
    this.toast('Import Successful', `Loaded ${newItems.length} rows from ${source}.`, 'ok');
  }

  async handleExcelFile(event) {
    const [file] = event.target.files ?? [];
    if (!file) return;
    try {
      const nameLower = file.name.toLowerCase();
      if (nameLower.endsWith('.csv')) {
        this.setStatus('Uploading and parsing CSV on the server...', 'info');
        const formData = new FormData();
        formData.append('file', file);

        const headers = {};
        if (this.state.token) headers.Authorization = `Bearer ${this.state.token}`;

        const response = await fetch(`${API_BASE}/solomon/import-csv`, {
          method: 'POST',
          headers,
          body: formData,
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => null);
          throw new Error(errData?.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        const incoming = Array.isArray(data?.customers) ? data.customers : [];
        if (!incoming.length) {
          throw new Error('No valid customer rows found in uploaded CSV.');
        }

        incoming.forEach((c) => {
          this.pushCustomer({
            name: c.name,
            address: c.address,
            lat: c.lat,
            lng: c.lng,
            demand: c.demand,
            ready: c.ready,
            due: c.due,
            service: c.service,
            isDepot: c.isDepot,
            priority: c.priority || 'Normal',
            skill: c.skill || 'None',
          });
        });

        this.setStatus(`Successfully imported ${incoming.length} customers from CSV file.`, 'ok');
        this.toast('Import Successful', `Loaded ${incoming.length} rows from CSV.`, 'ok');
      } else {
        if (typeof XLSX === 'undefined') throw new Error('SheetJS is not loaded yet');
        const buffer = await file.arrayBuffer();
        const workbook = XLSX.read(buffer, { type: 'array' });
        const firstSheet = workbook.SheetNames[0];
        const sheet = workbook.Sheets[firstSheet];
        const rows = XLSX.utils.sheet_to_json(sheet, { header: 1, defval: '' });
        const text = rows.map((cols) => cols.map((cell) => String(cell ?? '')).join('\t')).join('\n');
        await this.importCustomersFromText(text, 'excel file');
      }
    } catch (error) {
      this.toast('Import Failed', error.message, 'error');
      this.setStatus(`Unable to read file: ${error.message}`, 'error');
    } finally {
      if (this.el.excelInput) this.el.excelInput.value = '';
    }
  }

  normalizeHeaderKey(value) {
    return String(value || '')
      .trim()
      .toLowerCase()
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .replace(/[^a-z0-9]/g, '');
  }

  findHeaderIndex(cells, aliases) {
    const normalizedAliases = aliases.map((alias) => this.normalizeHeaderKey(alias));
    for (let i = 0; i < cells.length; i += 1) {
      if (normalizedAliases.includes(cells[i])) return i;
    }
    return -1;
  }

  detectHeaderMap(rows) {
    if (!Array.isArray(rows) || rows.length === 0) return null;

    const headerCells = rows[0].map((cell) => this.normalizeHeaderKey(cell));
    if (!headerCells.some((cell) => cell.length > 0)) return null;

    const map = {
      name: this.findHeaderIndex(headerCells, ['name', 'customer name', 'customer', 'client', 'store', 'shop']),
      address: this.findHeaderIndex(headerCells, ['address', 'addr', 'location', 'customer address', 'full address']),
      lat: this.findHeaderIndex(headerCells, ['lat', 'latitude', 'y', 'geo lat']),
      lng: this.findHeaderIndex(headerCells, ['lng', 'lon', 'long', 'longitude', 'x', 'geo lng', 'geo lon']),
      demand: this.findHeaderIndex(headerCells, ['demand', 'qty', 'quantity', 'load', 'order size', 'weight']),
      ready: this.findHeaderIndex(headerCells, [
        'ready',
        'readytime',
        'open',
        'tw start',
        'twstart',
        'earliest',
        'start',
      ]),
      due: this.findHeaderIndex(headerCells, [
        'due',
        'duedate',
        'duetime',
        'close',
        'tw end',
        'twend',
        'latest',
        'end',
        'deadline',
      ]),
      service: this.findHeaderIndex(headerCells, ['service', 'servicetime', 'svc', 'dwell', 'stoptime']),
    };

    const matchedCount = Object.values(map).filter((index) => index >= 0).length;
    const hasLocation = (map.lat >= 0 && map.lng >= 0) || map.address >= 0;
    if (!hasLocation || matchedCount < 2) return null;
    return map;
  }

  valueAt(cols, index, fallback = '') {
    if (!Array.isArray(cols) || index < 0 || index >= cols.length) return fallback;
    return String(cols[index] ?? '').trim();
  }

  readRowWithFallback(cols) {
    const base = cols.map((c) => String(c ?? '').trim());

    // Default format: name, address, lat, lng, demand, ready?, due?, service?
    let candidate = {
      name: base[0] || '',
      address: base[1] || '',
      latRaw: base[2] || '',
      lngRaw: base[3] || '',
      demandRaw: base[4] || '10',
      readyRaw: base[5] || '',
      dueRaw: base[6] || '',
      serviceRaw: base[7] || '',
    };

    const latA = Number(candidate.latRaw);
    const lngA = Number(candidate.lngRaw);
    if (Number.isFinite(latA) && Number.isFinite(lngA)) return candidate;

    // Alternate format with leading id: id, name, address, lat, lng, demand, ready?, due?, service?
    const candidateWithId = {
      name: base[1] || '',
      address: base[2] || '',
      latRaw: base[3] || '',
      lngRaw: base[4] || '',
      demandRaw: base[5] || '10',
      readyRaw: base[6] || '',
      dueRaw: base[7] || '',
      serviceRaw: base[8] || '',
    };
    const latB = Number(candidateWithId.latRaw);
    const lngB = Number(candidateWithId.lngRaw);
    if (Number.isFinite(latB) && Number.isFinite(lngB)) return candidateWithId;

    return candidate;
  }

  isDepotLabel(value) {
    const normalized = String(value || '')
      .trim()
      .toLowerCase();
    if (!normalized) return false;
    return ['depot', 'warehouse', 'hub', 'kho'].some((token) => normalized === token || normalized.includes(token));
  }

  async parseRowsToCustomers(rows) {
    const headerMap = this.detectHeaderMap(rows);
    const dataRows = headerMap ? rows.slice(1) : rows;
    const newItems = [];
    for (const cols of dataRows) {
      if (!Array.isArray(cols) || cols.length === 0) continue;

      const baseCells = cols.map((c) => String(c ?? '').trim());
      if (!baseCells.some((cell) => cell.length > 0)) continue;

      let name = '';
      let address = '';
      let latRaw = '';
      let lngRaw = '';
      let demandRaw = '10';
      let readyRaw = '';
      let dueRaw = '';
      let serviceRaw = '';

      if (headerMap) {
        name = this.valueAt(baseCells, headerMap.name, '');
        address = this.valueAt(baseCells, headerMap.address, '');
        latRaw = this.valueAt(baseCells, headerMap.lat, '');
        lngRaw = this.valueAt(baseCells, headerMap.lng, '');
        demandRaw = this.valueAt(baseCells, headerMap.demand, '10');
        readyRaw = this.valueAt(baseCells, headerMap.ready, '');
        dueRaw = this.valueAt(baseCells, headerMap.due, '');
        serviceRaw = this.valueAt(baseCells, headerMap.service, '');
      } else {
        const row = this.readRowWithFallback(baseCells);
        name = row.name;
        address = row.address;
        latRaw = row.latRaw;
        lngRaw = row.lngRaw;
        demandRaw = row.demandRaw;
        readyRaw = row.readyRaw;
        dueRaw = row.dueRaw;
        serviceRaw = row.serviceRaw;
      }

      let lat = Number(latRaw);
      let lng = Number(lngRaw);
      if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
        if (!address) continue;
        const geo = await this.tryGeocodeFromText(address);
        if (!geo) continue;
        lat = geo.lat;
        lng = geo.lng;
      }

      const parsedDemand = Number(demandRaw);
      const numericDemand = Number.isFinite(parsedDemand) ? parsedDemand : 0;
      const isDepot =
        this.isDepotLabel(name) ||
        this.isDepotLabel(address) ||
        (this.state.customers.length === 0 && newItems.length === 0 && numericDemand === 0);

      const readyNum = Number(readyRaw);
      const dueNum = Number(dueRaw);
      const serviceNum = Number(serviceRaw);
      const ready = Number.isFinite(readyNum) && readyNum >= 0 ? readyNum : 0;
      let due = Number.isFinite(dueNum) && dueNum > 0 ? dueNum : 1000;
      if (due <= ready) due = ready + 1;
      const serviceDefault = isDepot ? 0 : 10;
      const service = Number.isFinite(serviceNum) && serviceNum >= 0 ? serviceNum : serviceDefault;

      newItems.push({
        name: name || `Cust-${Date.now().toString().slice(-4)}`,
        address,
        lat,
        lng,
        demand: isDepot ? 0 : Number.isFinite(parsedDemand) ? parsedDemand : 10,
        ready,
        due,
        service,
        isDepot,
      });
    }
    return newItems;
  }

  async tryGeocodeFromText(address) {
    try {
      const geo = await this.request(`/geocode?q=${encodeURIComponent(address)}&limit=1`, { method: 'GET' });
      if (!geo.items || geo.items.length === 0) return null;
      return { lat: Number(geo.items[0].lat), lng: Number(geo.items[0].lng) };
    } catch {
      return null;
    }
  }

  async tryReverseGeocode(lat, lng) {
    try {
      const result = await this.request(
        `/reverse-geocode?lat=${encodeURIComponent(lat)}&lng=${encodeURIComponent(lng)}`,
        { method: 'GET' }
      );
      const shortAddress = result?.short_address?.trim();
      const fullAddress = result?.address?.trim();
      return shortAddress || fullAddress || '';
    } catch {
      return '';
    }
  }

  async addMapPoint(latlng) {
    const address =
      (await this.tryReverseGeocode(latlng.lat, latlng.lng)) ||
      `Lat ${latlng.lat.toFixed(5)}, Lng ${latlng.lng.toFixed(5)}`;
    const isFirst = this.state.customers.length === 0;
    this.pushCustomer({
      name: isFirst ? 'Depot' : `Pin-${this.state.customers.length}`,
      address,
      lat: latlng.lat,
      lng: latlng.lng,
      demand: 0,
      ready: 0,
      due: 1000,
      service: isFirst ? 0 : 10,
      isDepot: isFirst,
    });
    this.setStatus('Dropped a new delivery pin.', 'ok');
    this.toast('Pin Added', 'Point was added directly on the map.', 'ok');
  }

  focusTableInputField(field) {
    const input = this.tableInputRefs[field];
    if (!input) return;
    input.focus();
    input.select?.();
  }

  formatDraftCoord(value) {
    return Number.isFinite(value) ? Number(value).toFixed(5) : '-';
  }

  updateDraftCoordCells() {
    if (this.tableInputRefs.latCell)
      this.tableInputRefs.latCell.textContent = this.formatDraftCoord(this.tableInputDraft.lat);
    if (this.tableInputRefs.lngCell)
      this.tableInputRefs.lngCell.textContent = this.formatDraftCoord(this.tableInputDraft.lng);
  }

  async resolveTableInputAddress() {
    const address = this.tableInputDraft.address.trim();
    if (!address) {
      this.tableInputDraft.lat = null;
      this.tableInputDraft.lng = null;
      this.updateDraftCoordCells();
      return false;
    }
    const geo = await this.tryGeocodeFromText(address);
    if (!geo) {
      this.tableInputDraft.lat = null;
      this.tableInputDraft.lng = null;
      this.updateDraftCoordCells();
      return false;
    }
    this.tableInputDraft.lat = geo.lat;
    this.tableInputDraft.lng = geo.lng;
    this.updateDraftCoordCells();
    return true;
  }

  clearTableAddressSuggest() {
    this.tableAddressSuggest = [];
    this.tableAddressSuggestActive = -1;
    this.renderTableAddressSuggest();
  }

  scheduleTableAddressSuggest(query) {
    if (this.tableAddressSuggestTimer) {
      window.clearTimeout(this.tableAddressSuggestTimer);
      this.tableAddressSuggestTimer = 0;
    }

    const q = String(query || '').trim();
    if (q.length < 3) {
      this.clearTableAddressSuggest();
      return;
    }

    this.tableAddressSuggestTimer = window.setTimeout(async () => {
      this.tableAddressSuggestTimer = 0;
      try {
        const data = await this.request(`/geocode?q=${encodeURIComponent(q)}&limit=6`, { method: 'GET' });
        this.tableAddressSuggest = data.items || [];
        this.tableAddressSuggestActive = this.tableAddressSuggest.length > 0 ? 0 : -1;
      } catch {
        this.tableAddressSuggest = [];
        this.tableAddressSuggestActive = -1;
      }
      this.renderTableAddressSuggest();
    }, 220);
  }

  chooseTableAddressSuggestion(item) {
    if (!item) return;
    this.tableInputDraft.address = item.address || '';
    this.tableInputDraft.lat = Number(item.lat);
    this.tableInputDraft.lng = Number(item.lng);
    if (this.tableInputRefs.address) {
      this.tableInputRefs.address.value = this.tableInputDraft.address;
    }
    this.updateDraftCoordCells();
    this.clearTableAddressSuggest();
  }

  renderTableAddressSuggest() {
    const list = this.tableInputRefs.addressSuggest;
    if (!list) return;
    list.innerHTML = '';

    if (!Array.isArray(this.tableAddressSuggest) || this.tableAddressSuggest.length === 0) {
      list.classList.add('hidden');
      return;
    }

    this.tableAddressSuggest.forEach((item, idx) => {
      const li = document.createElement('li');
      li.textContent = item.address || '-';
      if (idx === this.tableAddressSuggestActive) li.classList.add('active');
      li.addEventListener('mousedown', (event) => event.preventDefault());
      li.addEventListener('click', () => {
        this.chooseTableAddressSuggestion(item);
        this.focusTableInputField('address');
      });
      list.appendChild(li);
    });
    list.classList.remove('hidden');
  }

  async handleTableInputEnter(field) {
    const order = ['name', 'address', 'demand', 'ready', 'due', 'service'];
    if (field === 'address') {
      await this.resolveTableInputAddress();
      this.focusTableInputField('demand');
      return;
    }
    if (field === 'service') {
      await this.submitTableInputRow();
      return;
    }
    const idx = order.indexOf(field);
    if (idx >= 0 && idx + 1 < order.length) {
      this.focusTableInputField(order[idx + 1]);
    }
  }

  moveTableInputFocus(field, direction) {
    const order = ['name', 'address', 'demand', 'ready', 'due', 'service'];
    const index = order.indexOf(field);
    if (index < 0) return;
    const nextIndex = Math.max(0, Math.min(order.length - 1, index + direction));
    if (nextIndex === index) return;
    this.focusTableInputField(order[nextIndex]);
  }

  resetTableInputDraft() {
    this.tableInputDraft = {
      name: '',
      address: '',
      demand: '0',
      lat: null,
      lng: null,
      ready: '0',
      due: '1000',
      service: '10',
    };
    this.clearTableAddressSuggest();
  }

  async submitTableInputRow() {
    const name = this.tableInputDraft.name.trim();
    const address = this.tableInputDraft.address.trim();
    const demandRaw = String(this.tableInputDraft.demand ?? '').trim();
    const readyRaw = String(this.tableInputDraft.ready ?? '').trim();
    const dueRaw = String(this.tableInputDraft.due ?? '').trim();
    const serviceRaw = String(this.tableInputDraft.service ?? '').trim();

    if (!name && !address && !demandRaw) {
      this.focusTableInputField('name');
      return;
    }

    const demand = Number(demandRaw || '0');
    if (!Number.isFinite(demand) || demand < 0) {
      this.toast('Invalid Demand', 'Demand must be a number >= 0.', 'error');
      this.focusTableInputField('demand');
      return;
    }

    const ready = Number(readyRaw || '0');
    if (!Number.isFinite(ready) || ready < 0) {
      this.toast('Invalid Ready', 'Ready time must be a number >= 0.', 'error');
      this.focusTableInputField('ready');
      return;
    }

    const due = Number(dueRaw || '1000');
    if (!Number.isFinite(due) || due <= ready) {
      this.toast('Invalid Due', 'Due time must be a number > Ready.', 'error');
      this.focusTableInputField('due');
      return;
    }

    const service = Number(serviceRaw || '0');
    if (!Number.isFinite(service) || service < 0) {
      this.toast('Invalid Service', 'Service time must be a number >= 0.', 'error');
      this.focusTableInputField('service');
      return;
    }

    if (!address) {
      this.toast('Missing Address', 'Please enter an address.', 'error');
      this.focusTableInputField('address');
      return;
    }

    if (!Number.isFinite(this.tableInputDraft.lat) || !Number.isFinite(this.tableInputDraft.lng)) {
      const found = await this.resolveTableInputAddress();
      if (!found) {
        this.toast('Geocode Failed', 'Could not resolve this address to coordinates.', 'error');
        this.focusTableInputField('address');
        return;
      }
    }

    const item = {
      name: name || `Cust-${this.state.customers.length}`,
      address,
      lat: this.tableInputDraft.lat,
      lng: this.tableInputDraft.lng,
      demand: Math.round(demand),
      ready,
      due,
      service,
    };

    this.tableInputVisible = true;
    this.resetTableInputDraft();
    this.pushCustomer(item);
    this.setStatus('Added customer from table input row.', 'ok');
    window.requestAnimationFrame(() => this.focusTableInputField('name'));
  }

  renderTableInputRow() {
    const tr = document.createElement('tr');
    tr.className = 'table-input-row';

    const pickCell = document.createElement('td');
    pickCell.textContent = '';
    tr.appendChild(pickCell);

    const idCell = document.createElement('td');
    idCell.textContent = '+';
    tr.appendChild(idCell);

    const createInputCell = (field, value, type = 'text') => {
      const td = document.createElement('td');
      if (field === 'address') td.classList.add('table-address-cell');
      const input = document.createElement('input');
      input.className = 'table-inline-input';
      input.type = type;
      input.value = value;
      if (field === 'demand') {
        input.min = '0';
        input.step = '1';
      }
      if (field === 'ready' || field === 'due' || field === 'service') {
        input.min = '0';
        input.step = 'any';
      }
      input.addEventListener('input', () => {
        this.tableInputDraft[field] = input.value;
        if (field === 'address') {
          this.tableInputDraft.lat = null;
          this.tableInputDraft.lng = null;
          this.updateDraftCoordCells();
          this.scheduleTableAddressSuggest(input.value);
        }
      });
      input.addEventListener('keydown', (event) => {
        if (field === 'address' && event.key === 'ArrowDown' && this.tableAddressSuggest.length > 0) {
          event.preventDefault();
          this.tableAddressSuggestActive = Math.min(
            this.tableAddressSuggest.length - 1,
            this.tableAddressSuggestActive + 1
          );
          this.renderTableAddressSuggest();
          return;
        }
        if (field === 'address' && event.key === 'ArrowUp' && this.tableAddressSuggest.length > 0) {
          event.preventDefault();
          this.tableAddressSuggestActive = Math.max(0, this.tableAddressSuggestActive - 1);
          this.renderTableAddressSuggest();
          return;
        }
        if (field === 'address' && event.key === 'Tab' && !event.shiftKey) {
          if (this.tableAddressSuggest.length > 0 && this.tableAddressSuggestActive >= 0) {
            event.preventDefault();
            this.chooseTableAddressSuggestion(this.tableAddressSuggest[this.tableAddressSuggestActive]);
            this.focusTableInputField('demand');
            return;
          }
          this.clearTableAddressSuggest();
        }
        if (event.key === 'Enter') {
          event.preventDefault();
          if (field === 'address' && this.tableAddressSuggest.length > 0 && this.tableAddressSuggestActive >= 0) {
            this.chooseTableAddressSuggestion(this.tableAddressSuggest[this.tableAddressSuggestActive]);
            this.focusTableInputField('demand');
            return;
          }
          this.handleTableInputEnter(field);
          return;
        }
        if (event.key === 'ArrowRight') {
          event.preventDefault();
          this.moveTableInputFocus(field, 1);
          return;
        }
        if (event.key === 'ArrowLeft') {
          event.preventDefault();
          this.moveTableInputFocus(field, -1);
        }
      });
      if (field === 'address') {
        input.addEventListener('blur', () => {
          window.setTimeout(() => {
            this.clearTableAddressSuggest();
            this.resolveTableInputAddress();
          }, 100);
        });
        input.addEventListener('focus', () => {
          this.renderTableAddressSuggest();
        });
      }
      td.appendChild(input);
      if (field === 'address') {
        const suggest = document.createElement('ul');
        suggest.className = 'table-address-suggest hidden';
        td.appendChild(suggest);
        this.tableInputRefs.addressSuggest = suggest;
      }
      this.tableInputRefs[field] = input;
      return td;
    };

    tr.appendChild(createInputCell('name', this.tableInputDraft.name));
    tr.appendChild(createInputCell('address', this.tableInputDraft.address));

    const latCell = document.createElement('td');
    latCell.className = 'table-inline-readonly';
    latCell.textContent = this.formatDraftCoord(this.tableInputDraft.lat);
    this.tableInputRefs.latCell = latCell;
    tr.appendChild(latCell);

    const lngCell = document.createElement('td');
    lngCell.className = 'table-inline-readonly';
    lngCell.textContent = this.formatDraftCoord(this.tableInputDraft.lng);
    this.tableInputRefs.lngCell = lngCell;
    tr.appendChild(lngCell);

    tr.appendChild(createInputCell('demand', this.tableInputDraft.demand, 'number'));
    tr.appendChild(createInputCell('ready', this.tableInputDraft.ready, 'number'));
    tr.appendChild(createInputCell('due', this.tableInputDraft.due, 'number'));
    tr.appendChild(createInputCell('service', this.tableInputDraft.service, 'number'));

    this.el.customerRows.appendChild(tr);
  }

  pushCustomer(item) {
    const id = this.state.customers.length;
    const isDepot = Boolean(item.isDepot);
    const ready = Number.isFinite(Number(item.ready)) ? Number(item.ready) : 0;
    let due = Number.isFinite(Number(item.due)) ? Number(item.due) : 1000;
    if (due <= ready) due = ready + 1;
    const service = Number.isFinite(Number(item.service)) ? Number(item.service) : isDepot ? 0 : 10;
    this.state.customers.push({ ...item, id, isDepot, ready, due, service });
    this.selectedCustomerIds.clear();
    this.renderCustomers();
    this.mapController.renderMarkers();
  }

  renderCustomers() {
    this.tableInputRefs = {};
    this.el.customerRows.innerHTML = '';
    this.state.customers.forEach((c) => {
      const tr = document.createElement('tr');
      tr.dataset.customerId = String(c.id);
      if (this.selectedCustomerIds.has(c.id)) tr.classList.add('table-row-selected');

      tr.addEventListener('click', (event) => {
        const interactive = event.target.closest('input, button, a, [data-editable="true"]');
        if (interactive) return;
        this.toggleCustomerSelection(c.id);
      });

      const pick = document.createElement('td');
      pick.className = 'table-pick-cell';
      const selected = this.selectedCustomerIds.has(c.id);
      pick.textContent = selected ? '✓' : '○';
      if (selected) pick.classList.add('is-selected');
      tr.appendChild(pick);

      const fmtTime = (v, fallback) => {
        const n = Number(v);
        if (!Number.isFinite(n)) return String(fallback);
        return Number.isInteger(n) ? String(n) : n.toFixed(1);
      };
      const values = [
        { field: 'id', value: String(c.id), editable: false },
        { field: 'name', value: c.name || '', editable: true },
        { field: 'address', value: c.address || '-', editable: true },
        { field: 'lat', value: Number(c.lat).toFixed(5), editable: false },
        { field: 'lng', value: Number(c.lng).toFixed(5), editable: false },
        { field: 'demand', value: String(c.demand), editable: true },
        { field: 'ready', value: fmtTime(c.ready, 0), editable: true },
        { field: 'due', value: fmtTime(c.due, 1000), editable: true },
        { field: 'service', value: fmtTime(c.service, c.isDepot ? 0 : 10), editable: true },
        { field: 'priority', value: c.priority || 'Normal', editable: true },
        { field: 'skill', value: c.skill || 'None', editable: true },
      ];

      values.forEach(({ field, value, editable }) => {
        const td = document.createElement('td');
        td.dataset.field = field;
        if (editable) {
          td.dataset.editable = 'true';
          td.classList.add('cell-editable');
          td.title = 'Double-click to edit';
        }

        if (field === 'name' && c.isDepot) {
          td.classList.add('customer-name-cell');
          const label = document.createElement('span');
          label.className = 'customer-name-label';
          label.textContent = value || 'Depot';

          const badge = document.createElement('span');
          badge.className = 'depot-badge';
          badge.textContent = 'DEPOT';

          td.appendChild(label);
          td.appendChild(badge);
        } else if (field === 'priority' && !c.isDepot) {
          td.textContent = '';
          const badge = document.createElement('span');
          badge.className = 'priority-badge';
          badge.textContent = value;
          if (value === 'High') {
            badge.style =
              'background: rgba(239, 68, 68, 0.1); color: var(--danger); border: 1px solid rgba(239,68,68,0.2); font-weight: 700; padding: 2px 6px; border-radius: 4px; font-size: 10px; display: inline-block; text-align: center; width: 55px;';
          } else if (value === 'Low') {
            badge.style =
              'background: rgba(107, 114, 128, 0.1); color: var(--text-muted); border: 1px solid rgba(107,114,128,0.2); font-weight: 500; padding: 2px 6px; border-radius: 4px; font-size: 10px; display: inline-block; text-align: center; width: 55px;';
          } else {
            badge.style =
              'background: rgba(59, 130, 246, 0.1); color: var(--primary); border: 1px solid rgba(59,130,246,0.2); font-weight: 600; padding: 2px 6px; border-radius: 4px; font-size: 10px; display: inline-block; text-align: center; width: 55px;';
          }
          td.appendChild(badge);
        } else if (field === 'skill' && !c.isDepot) {
          td.innerHTML = createSkillBadge(value);
        } else {
          td.textContent = value;
        }
        tr.appendChild(td);
      });

      this.el.customerRows.appendChild(tr);
    });
    if (this.tableInputVisible) this.renderTableInputRow();
    this.showEmptyStates();
  }

  showEmptyStates() {
    const hasCustomers = this.state.customers.length > 0;
    const hasResult = Boolean(this.state.lastResult);

    this.el.tableEmpty?.classList.toggle('hidden', hasCustomers || this.tableInputVisible);
    this.el.mapEmptyDdqn?.classList.toggle('hidden', hasResult);
    this.el.mapEmptyAlns?.classList.toggle('hidden', hasResult);

    if (!hasResult) {
      this.mapController.ddqnVehicleLayer?.clearLayers();
      this.mapController.alnsVehicleLayer?.clearLayers();
    }
  }

  async submitJob() {
    try {
      if (!this.state.token) {
        this.toast('Not Logged In', 'Please log in before running the model.', 'error');
        return;
      }
      if (this.state.customers.length < 2) {
        this.setStatus('At least depot and 1 customer are required.', 'error');
        return;
      }

      this.showLoading(true);
      this.runSession.token += 1;
      this.runSession.cancelled = false;
      this.runSession.abortController?.abort();
      this.runSession.abortController = new AbortController();
      this.setStatus('Submitting optimization job to background queue...');
      this.toast('Processing', 'Job is being queued and will run in the background.', 'ok');

      const payload = {
        mode: this.state.mode,
        dataset: this.state.mode === 'sample' ? (this.el.datasetSelect?.value || 'demo') : '',
        preset: this.state.preset || 'fast',
        pretrained_transfer: Boolean(this.state.pretrainedTransfer),
        use_gnn: Boolean(this.state.useGnn !== false),
        fleet: { vehicles: this.state.vehicles, capacity: this.state.capacity },
        customers: this.state.customers,
      };
      this.state.lastRunFleet = { ...payload.fleet };

      const submit = await this.request('/jobs', {
        method: 'POST',
        body: JSON.stringify(payload),
        signal: this.runSession.abortController.signal,
      });

      if (typeof window.vrptwTrack === 'function') {
        window.vrptwTrack('job_submit', {
          mode: this.state.mode,
          customers: this.state.customers.length,
          vehicles: this.state.vehicles,
          capacity: this.state.capacity,
        });
      }

      await firebaseService.saveJobStart(submit.job_id, {
        mode: this.state.mode,
        fleet: payload.fleet,
        customerCount: this.state.customers.length,
        customers: this.state.customers,
      });

      const pollLimitMs = this.estimatePollTimeoutMs(this.state.customers.length);
      await this.pollJob(submit.job_id, pollLimitMs, this.runSession.token);
    } catch (error) {
      if (this.runSession.cancelled || error?.name === 'AbortError') {
        return;
      }

      // ── Client-side fallback for demo/guest mode ──────────────────
      // If the backend is down or returns 503, solve entirely in-browser
      // so the demo experience remains seamless.
      const isGuest = this.state.role === 'guest' || this.state.token === 'demo-guest';
      if (isGuest) {
        try {
          this.setStatus('Backend unavailable — running client-side solver...', 'warn');
          this.toast('Local Solver', 'Backend is offline. Running in-browser ALNS solver.', 'ok');
          const demoResult = solveDemo(
            this.state.customers,
            this.state.vehicles,
            this.state.capacity,
            this.state.fleet
          );
          this.state.lastResult = demoResult;
          this.paintResult();
          await this.completeLoading();
          this.setStatus('Client-side optimization complete.', 'ok');
          this.toast('Model Completed', 'Results rendered using the in-browser solver.', 'ok');
          this.showEmptyStates();
          return;
        } catch (fallbackErr) {
          console.error('Client-side solver also failed:', fallbackErr);
        }
      }

      const raw = this.parseApiError(error);
      const adjusted = this.autoAdjustVehiclesForInfeasible(raw);
      const friendly = this.formatRunError(raw);
      if (adjusted?.changed) {
        const msg =
          adjusted.target >= adjusted.max
            ? `Auto-adjusted Vehicles to ${adjusted.target} (slider max). If still infeasible, increase Capacity.`
            : `Auto-adjusted Vehicles: ${adjusted.current} -> ${adjusted.target}. Click Run Model again.`;
        this.toast('Vehicles Auto-Adjusted', msg, 'ok');
      }
      this.setStatus(`Submit error: ${friendly}`, 'error');
      this.toast('Submit Failed', friendly, 'error');
      await this.showLoadingError(friendly);
    } finally {
      this.runSession.abortController = null;
    }
  }

  estimatePollTimeoutMs(customerCount) {
    const count = Math.max(0, Number(customerCount) || 0);
    const baseline = 600000;
    const scaled = baseline + count * 6000;
    return Math.min(1800000, Math.max(baseline, scaled));
  }

  async pollJob(jobId, timeoutMs = 180000, sessionToken = 0) {
    const startedAt = Date.now();
    while (Date.now() - startedAt < timeoutMs) {
      if (this.runSession.cancelled || sessionToken !== this.runSession.token) return;
      const data = await this.request(`/jobs/${jobId}`, {
        method: 'GET',
        signal: this.runSession.abortController?.signal,
      });
      if (this.runSession.cancelled || sessionToken !== this.runSession.token) return;

      const phase = data?.debug?.phase || data.status;
      this.loadingAnim.backendPhase = phase;

      const events = data?.debug?.events || [];
      events.forEach((evt) => {
        if (evt && evt.message) {
          this.addConsoleLog(evt.message);
        }
      });

      if (events.length > 0) {
        this.loadingAnim.backendMessage = events[events.length - 1].message;
      }

      if (phase === 'queued') {
        this.setLoadingSubtitle('Backend has accepted the job. Waiting for a worker thread.');
        this.setLoadingBackend('queued', 'warn');
      } else if (phase === 'processing') {
        this.setLoadingSubtitle('Worker reserved. Preparing solver inputs...');
        this.setLoadingBackend('processing', 'ok');
      } else if (phase === 'matrix') {
        this.setLoadingSubtitle('Computing distance matrix for route optimization...');
        this.setLoadingBackend('building matrix', 'ok');
      } else if (phase === 'solving') {
        this.setLoadingSubtitle('Running parallel DDQN-ALNS hybrid and ALNS baseline solvers...');
        this.setLoadingBackend('solving model', 'ok');
      }

      if (data.status === 'done') {
        if (this.runSession.cancelled || sessionToken !== this.runSession.token) return;
        this.state.lastResult = data.result;
        this.paintResult();
        await firebaseService.saveJobResult(jobId, data.result);
        await this.completeLoading();
        this.setStatus('Received optimization results from backend.', 'ok');
        this.toast('Model Completed', 'Results have been rendered on the dashboard.', 'ok');
        this.showEmptyStates();
        return;
      }
      if (data.status === 'failed') {
        throw new Error(data.error || 'Job failed');
      }
      await new Promise((resolve) => setTimeout(resolve, 1400));
    }
    const seconds = Math.round(timeoutMs / 1000);
    throw new Error(
      `Job timeout after ${seconds}s. Backend may still be running; try increasing Vehicles/Capacity or reducing customer count.`
    );
  }

  paintResult() {
    const result = this.state.lastResult;
    if (!result) return;

    this.mapController.clearRoutes();
    const routeCapacity = Number(this.state.lastRunFleet?.capacity ?? this.state.capacity);

    const colors = {
      ddqn: '#0b8a65',
      alns: '#2563eb',
      ortools: '#e11d48',
      hybrid_fixed: '#d97706',
      hybrid_ddqn: '#7c3aed',
      hybrid_ddqn_transfer_rc1: '#0284c7',
      hybrid_ddqn_transfer_dr: '#4f46e5',
      hybrid: '#0b8a65',
    };

    for (const algoName in result) {
      const color = colors[algoName] || '#6b7280';
      this.mapController.renderAlgoRoutes(result[algoName], algoName, color, routeCapacity);
    }

    if (result.ddqn && result.alns) {
      this.mapController.renderAlnsOnlySegments(result.ddqn, result.alns);
    }

    this.mapController.initSimulation(result);

    // Dynamic map view options in the KPI strip's overlay dropdown
    const overlaySelect = document.getElementById('map-view-select');
    if (overlaySelect) {
      const labels = {
        ddqn: 'Hybrid DDQN (Transfer)',
        alns: 'ALNS Base',
        ortools: 'OR-Tools',
        hybrid_fixed: 'Hybrid Fixed',
        hybrid_ddqn: 'Hybrid DDQN (Random)',
        hybrid_ddqn_transfer_rc1: 'Hybrid DDQN (RC1)',
        hybrid_ddqn_transfer_dr: 'Hybrid DDQN (DR)',
      };

      let html = '';
      const algoNames = Object.keys(result);
      const current = this.mapController.currentView;
      const currentSelected = algoNames.includes(current) ? current : algoNames[0];

      algoNames.forEach((algoName) => {
        const label = labels[algoName] || algoName;
        html += `<option value="${algoName}">${label}</option>`;
      });
      overlaySelect.innerHTML = html;
      overlaySelect.value = currentSelected;

      // Assigned rather than added: the select survives re-renders, so stacking
      // listeners here would fire switchView once per past solve.
      overlaySelect.onchange = (e) => {
        this.mapController.switchView(e.target.value);
      };
    }

    const initialView = result.ddqn ? 'ddqn' : Object.keys(result)[0];
    this.mapController.switchView(initialView);

    this.mapController
      .fetchRoadGeometries(result)
      .then(() => {
        this.setStatus('Road geometry loaded — routes now follow actual roads.', 'ok');
      })
      .catch((err) => {
        console.warn('OSRM road geometry fetch failed, keeping straight-line routes:', err);
      });

    let maxTime = 120;
    const findMax = (algo) => {
      if (algo && algo.routes) {
        algo.routes.forEach((r) => {
          if (r.schedule && r.schedule.length > 0) {
            const endStep = r.schedule[r.schedule.length - 1];
            if (endStep && endStep.arrival > maxTime) {
              maxTime = endStep.arrival;
            }
          }
        });
      }
    };
    for (const algoName in result) {
      findMax(result[algoName]);
    }
    this.simulationController.start(maxTime);
    this.ganttController?.render(result, initialView);

    this.updateComparisonMetrics(initialView);
    this.showEmptyStates();
  }

  updateKpiCards(activeAlgo) {
    const result = this.state.lastResult;
    if (!result) return;

    const controlAlgo = 'alns';
    const testAlgo = activeAlgo;

    const testResult = result[testAlgo];
    const controlResult = result[controlAlgo];

    if (!testResult || !controlResult) return;

    const labels = {
      ddqn: 'Hybrid DDQN (Transfer)',
      alns: 'ALNS Base',
      ortools: 'OR-Tools',
      hybrid_fixed: 'Hybrid Fixed',
      hybrid_ddqn: 'Hybrid DDQN (Random)',
      hybrid_ddqn_transfer_rc1: 'Hybrid DDQN (RC1)',
      hybrid_ddqn_transfer_dr: 'Hybrid DDQN (DR)',
    };
    const testLabel = labels[testAlgo] || testAlgo;

    document.querySelectorAll('.kpi-card').forEach((card) => {
      const labelsList = card.querySelectorAll('.kpi-label');
      if (labelsList.length >= 2) {
        labelsList[0].textContent = testLabel;
        labelsList[1].textContent = 'ALNS Base';
      }
    });

    const testDist = Number(testResult.total_distance_km || 0);
    const controlDist = Number(controlResult.total_distance_km || 0);
    const gapVal = controlDist > 0 ? ((testDist - controlDist) / controlDist) * 100 : 0;

    const kpiGap = document.getElementById('kpi-gap');
    if (kpiGap) {
      kpiGap.textContent = `${gapVal >= 0 ? '+' : ''}${gapVal.toFixed(2)}%`;
      if (gapVal < 0) {
        kpiGap.className = 'kpi-value highlight-emerald';
      } else if (gapVal > 0) {
        kpiGap.className = 'kpi-value highlight-rose';
      } else {
        kpiGap.className = 'kpi-value highlight-blue';
      }
    }

    const gapTitle = document.querySelector('.kpi-card:nth-child(1) .kpi-title');
    if (gapTitle) {
      gapTitle.textContent = `Gap vs ALNS Base`;
    }

    const distTest = document.getElementById('kpi-dist-ddqn');
    const distControl = document.getElementById('kpi-dist-alns');
    if (distTest) distTest.textContent = testDist.toFixed(2);
    if (distControl) distControl.textContent = controlDist.toFixed(2);

    const vehTest = document.getElementById('kpi-veh-ddqn');
    const vehControl = document.getElementById('kpi-veh-alns');
    if (vehTest) vehTest.textContent = testResult.vehicles_used ?? 0;
    if (vehControl) vehControl.textContent = controlResult.vehicles_used ?? 0;

    const timeTest = document.getElementById('kpi-time-ddqn');
    const timeControl = document.getElementById('kpi-time-alns');
    if (timeTest) timeTest.textContent = `${Number(testResult.runtime_sec || 0).toFixed(2)}s`;
    if (timeControl) timeControl.textContent = `${Number(controlResult.runtime_sec || 0).toFixed(2)}s`;
  }

  updateComparisonMetrics(view) {
    const result = this.state.lastResult;
    if (!result) return;

    const controlAlgo = 'alns';
    const testAlgo = view;

    const testResult = result[testAlgo];
    const controlResult = result[controlAlgo];

    if (!testResult || !controlResult) return;

    const routeCapacity = Number(this.state.lastRunFleet?.capacity ?? this.state.capacity);

    this.updateKpiCards(view);

    this.updateCompareMetric({
      card: this.el.metricRuntimeCard,
      ddqnNode: this.el.metricRuntimeDdqn,
      alnsNode: this.el.metricRuntimeAlns,
      deltaNode: this.el.metricRuntimeDelta,
      barDdqn: this.el.metricRuntimeBarDdqn,
      barAlns: this.el.metricRuntimeBarAlns,
      ddqn: testResult.runtime_sec,
      alns: controlResult.runtime_sec,
      unit: 's',
      decimals: 2,
      lowerIsBetter: true,
    });

    this.updateCompareMetric({
      card: this.el.metricDistanceCard,
      ddqnNode: this.el.metricDistanceDdqn,
      alnsNode: this.el.metricDistanceAlns,
      deltaNode: this.el.metricDistanceDelta,
      barDdqn: this.el.metricDistanceBarDdqn,
      barAlns: this.el.metricDistanceBarAlns,
      ddqn: testResult.total_distance_km,
      alns: controlResult.total_distance_km,
      unit: 'km',
      decimals: 2,
      lowerIsBetter: true,
    });

    this.updateCompareMetric({
      card: this.el.metricVehiclesCard,
      ddqnNode: this.el.metricVehiclesDdqn,
      alnsNode: this.el.metricVehiclesAlns,
      deltaNode: this.el.metricVehiclesDelta,
      barDdqn: this.el.metricVehiclesBarDdqn,
      barAlns: this.el.metricVehiclesBarAlns,
      ddqn: testResult.vehicles_used,
      alns: controlResult.vehicles_used,
      unit: '',
      decimals: 0,
      lowerIsBetter: true,
    });

    this.updateLoadInsight(testResult, controlResult, routeCapacity);
  }

  resetResultOutputs() {
    this.state.lastResult = null;
    this.mapController.clearRoutes();
    this.simulationController.showPanels(false);
    this.ganttController?.render(null);

    const setText = (node, value) => {
      if (node) node.textContent = value;
    };
    const setWidth = (node, value) => {
      if (node) node.style.width = value;
    };

    setText(this.el.metricRuntimeDdqn, '0.00s');
    setText(this.el.metricRuntimeAlns, '0.00s');
    setText(this.el.metricRuntimeDelta, 'Results are tied');

    setText(this.el.metricDistanceDdqn, '0.00km');
    setText(this.el.metricDistanceAlns, '0.00km');
    setText(this.el.metricDistanceDelta, 'Results are tied');

    setText(this.el.metricVehiclesDdqn, '0');
    setText(this.el.metricVehiclesAlns, '0');
    setText(this.el.metricVehiclesDelta, 'Results are tied');

    setWidth(this.el.metricDistanceBarDdqn, '0%');
    setWidth(this.el.metricDistanceBarAlns, '0%');
    setWidth(this.el.metricVehiclesBarDdqn, '0%');
    setWidth(this.el.metricVehiclesBarAlns, '0%');

    setText(this.el.metricLoadDdqn, '0.0%');
    setText(this.el.metricLoadAlns, '0.0%');
    setText(this.el.metricLoadDelta, 'Balanced');
    setWidth(this.el.metricLoadBarDdqn, '0%');
    setWidth(this.el.metricLoadBarAlns, '0%');
    if (this.el.metricLoadDonutDdqn) this.el.metricLoadDonutDdqn.style.setProperty('--p', '0%');
    if (this.el.metricLoadDonutAlns) this.el.metricLoadDonutAlns.style.setProperty('--p', '0%');
    setText(this.el.metricLoadDonutDdqnLabel, '0%');
    setText(this.el.metricLoadDonutAlnsLabel, '0%');
    if (this.el.metricLoadDdqnState) {
      this.el.metricLoadDdqnState.className = 'load-state';
      this.el.metricLoadDdqnState.textContent = 'No data';
    }
    if (this.el.metricLoadAlnsState) {
      this.el.metricLoadAlnsState.className = 'load-state';
      this.el.metricLoadAlnsState.textContent = 'No data';
    }

    [this.el.metricRuntimeCard, this.el.metricDistanceCard, this.el.metricVehiclesCard, this.el.metricLoadCard].forEach(
      (card) => {
        if (card) card.dataset.winner = 'tie';
      }
    );

    this.showEmptyStates();
  }

  classifyLoadState(value) {
    if (!Number.isFinite(value)) return { key: '', label: 'No data' };
    if (value > 95) return { key: 'critical', label: 'Critical load' };
    if (value >= 80) return { key: 'near', label: 'Near full' };
    return { key: 'safe', label: 'Safe load' };
  }

  computeAlgoUtilizationPercent(algo, capacity) {
    const routes = Array.isArray(algo?.routes) ? algo.routes : [];
    const cap = Number(capacity);
    if (!Number.isFinite(cap) || cap <= 0 || routes.length === 0) return 0;

    const totalLoad = routes.reduce((sum, route) => sum + Number(route?.load || 0), 0);
    const totalCap = routes.length * cap;
    if (totalCap <= 0) return 0;
    return (totalLoad / totalCap) * 100;
  }

  updateLoadInsight(ddqnAlgo, alnsAlgo, capacity) {
    const ddqnPct = this.computeAlgoUtilizationPercent(ddqnAlgo, capacity);
    const alnsPct = this.computeAlgoUtilizationPercent(alnsAlgo, capacity);
    const ddqnClamped = Math.max(0, Math.min(100, ddqnPct));
    const alnsClamped = Math.max(0, Math.min(100, alnsPct));

    if (this.el.metricLoadDdqn) this.el.metricLoadDdqn.textContent = `${ddqnPct.toFixed(1)}%`;
    if (this.el.metricLoadAlns) this.el.metricLoadAlns.textContent = `${alnsPct.toFixed(1)}%`;
    if (this.el.metricLoadBarDdqn) this.el.metricLoadBarDdqn.style.width = `${ddqnClamped}%`;
    if (this.el.metricLoadBarAlns) this.el.metricLoadBarAlns.style.width = `${alnsClamped}%`;
    if (this.el.metricLoadDonutDdqn) this.el.metricLoadDonutDdqn.style.setProperty('--p', `${ddqnClamped}%`);
    if (this.el.metricLoadDonutAlns) this.el.metricLoadDonutAlns.style.setProperty('--p', `${alnsClamped}%`);
    if (this.el.metricLoadDonutDdqnLabel) this.el.metricLoadDonutDdqnLabel.textContent = `${ddqnClamped.toFixed(0)}%`;
    if (this.el.metricLoadDonutAlnsLabel) this.el.metricLoadDonutAlnsLabel.textContent = `${alnsClamped.toFixed(0)}%`;

    const ddqnState = this.classifyLoadState(ddqnPct);
    const alnsState = this.classifyLoadState(alnsPct);
    if (this.el.metricLoadDdqnState) {
      this.el.metricLoadDdqnState.className = `load-state ${ddqnState.key}`.trim();
      this.el.metricLoadDdqnState.textContent = ddqnState.label;
    }
    if (this.el.metricLoadAlnsState) {
      this.el.metricLoadAlnsState.className = `load-state ${alnsState.key}`.trim();
      this.el.metricLoadAlnsState.textContent = alnsState.label;
    }

    if (this.el.metricLoadDelta) {
      const diff = Math.abs(ddqnPct - alnsPct);
      if (diff < 0.05) {
        this.el.metricLoadDelta.textContent = 'Balanced';
      } else {
        const winner = ddqnPct > alnsPct ? 'DDQN' : 'ALNS';
        this.el.metricLoadDelta.textContent = `${winner} +${diff.toFixed(1)}%`;
      }
    }

    if (this.el.metricLoadCard) {
      if (Math.abs(ddqnPct - alnsPct) < 0.05) this.el.metricLoadCard.dataset.winner = 'tie';
      else this.el.metricLoadCard.dataset.winner = ddqnPct > alnsPct ? 'ddqn' : 'alns';
    }
  }

  updateCompareMetric({
    card,
    ddqnNode,
    alnsNode,
    deltaNode,
    barDdqn,
    barAlns,
    ddqn,
    alns,
    unit,
    decimals,
    lowerIsBetter,
  }) {
    const ddqnValue = Number(ddqn);
    const alnsValue = Number(alns);
    const tieThreshold = 1e-9;
    const diff = alnsValue - ddqnValue;
    const absDiff = Math.abs(diff);

    let winner = 'tie';
    if (absDiff > tieThreshold) {
      if (lowerIsBetter) {
        winner = ddqnValue < alnsValue ? 'ddqn' : 'alns';
      } else {
        winner = ddqnValue > alnsValue ? 'ddqn' : 'alns';
      }
    }

    if (ddqnNode) ddqnNode.textContent = this.formatMetricValue(ddqnValue, unit, decimals);
    if (alnsNode) alnsNode.textContent = this.formatMetricValue(alnsValue, unit, decimals);

    if (deltaNode) {
      if (winner === 'tie') {
        deltaNode.textContent = 'Results are tied';
      } else {
        const betterLabel = winner.toUpperCase();
        deltaNode.textContent = `${betterLabel} is better by ${this.formatMetricValue(absDiff, unit, decimals)}`;
      }
    }

    if (card) card.dataset.winner = winner;

    const epsilon = 1e-9;
    if (lowerIsBetter) {
      const min = Math.min(ddqnValue, alnsValue) + epsilon;
      const ddqnScore = min / (ddqnValue + epsilon);
      const alnsScore = min / (alnsValue + epsilon);
      if (barDdqn) barDdqn.style.width = `${30 + ddqnScore * 70}%`;
      if (barAlns) barAlns.style.width = `${30 + alnsScore * 70}%`;
    } else {
      const max = Math.max(ddqnValue, alnsValue) + epsilon;
      const ddqnScore = ddqnValue / max;
      const alnsScore = alnsValue / max;
      if (barDdqn) barDdqn.style.width = `${30 + ddqnScore * 70}%`;
      if (barAlns) barAlns.style.width = `${30 + alnsScore * 70}%`;
    }
  }

  formatMetricValue(value, unit, decimals) {
    const fixed = Number(value).toFixed(decimals);
    return `${fixed}${unit}`;
  }

  clearConsoleLog() {
    if (this.el.loadingConsole) {
      this.el.loadingConsole.innerHTML = '';
    }
    this.loadingAnim.loggedEvents = new Set();
  }

  addConsoleLog(message) {
    if (!this.el.loadingConsole || !message) return;
    if (!this.loadingAnim.loggedEvents) {
      this.loadingAnim.loggedEvents = new Set();
    }
    if (this.loadingAnim.loggedEvents.has(message)) return;
    this.loadingAnim.loggedEvents.add(message);

    const placeholder = this.el.loadingConsole.querySelector('.placeholder');
    if (placeholder) {
      placeholder.remove();
    }

    const elapsedMs = Date.now() - (this.loadingAnim.runStartedAt || Date.now());
    const elapsedSec = (elapsedMs / 1000).toFixed(1);

    const line = document.createElement('div');
    line.className = 'loading-console-line';
    line.innerHTML = `<span style="color: #64748b; margin-right: 6px;">[${elapsedSec}s]</span> ${message}`;

    this.el.loadingConsole.appendChild(line);
    this.el.loadingConsole.scrollTop = this.el.loadingConsole.scrollHeight;
  }

  async showLoadingError(message) {
    this.stopLoadingProgress();
    this.loadingAnim.backendPhase = 'failed';
    this.addConsoleLog(`Error: ${message}`);
    this.setLoadingProgress(this.loadingAnim.progress, 'Optimization Failed', message, 0, 'idle');
    this.setLoadingBackend('failed', 'error');
    if (this.el.loadingCard) {
      this.el.loadingCard.setAttribute('data-phase', 'failed');
      this.el.loadingCard.classList.add('loading-card-shake');
    }
    await new Promise((resolve) => setTimeout(resolve, 3200));
    if (this.el.loadingCard) {
      this.el.loadingCard.classList.remove('loading-card-shake');
    }
    this.el.loading?.classList.add('hidden');
  }

  setLoadingProgress(value, title = null, subtitle = null, normalizedSpeed = 0, algo = 'idle') {
    const clamped = Math.max(0, Math.min(100, Number(value) || 0));
    this.loadingAnim.progress = clamped;

    if (title && this.el.loadingTitle) {
      this.el.loadingTitle.textContent = title;
    }
    if (this.el.loadingPhase && typeof subtitle === 'string') {
      this.el.loadingPhase.textContent = subtitle;
    }

    if (this.el.loadingCard) {
      this.el.loadingCard.style.setProperty('--truck-speed', Number(normalizedSpeed || 0).toFixed(3));
      this.el.loadingCard.dataset.algo = algo;

      const activePhase = this.loadingAnim.backendPhase || 'queued';
      this.el.loadingCard.setAttribute('data-phase', activePhase);

      this.el.loadingCard.classList.remove('is-slow', 'is-medium', 'is-fast');
      if (normalizedSpeed > 0.67) this.el.loadingCard.classList.add('is-fast');
      else if (normalizedSpeed > 0.34) this.el.loadingCard.classList.add('is-medium');
      else this.el.loadingCard.classList.add('is-slow');
    }

    if (this.el.loadingTrackFill) {
      this.el.loadingTrackFill.style.setProperty('--loading-progress', `${clamped}%`);
    }
    if (this.el.loadingTruck) {
      this.el.loadingTruck.style.setProperty('--loading-progress', `${clamped}%`);
    }
    if (this.el.loadingPercent) {
      this.el.loadingPercent.textContent = `${Math.round(clamped)}%`;
    }
  }

  showLoading(show) {
    if (!this.el.loading) return;
    if (show) {
      this.el.loading.classList.remove('hidden', 'loading--minimized');
      this.el.loadingLauncher?.classList.add('hidden');
      this.startLoadingProgress();
    } else {
      this.hideLoadingImmediate();
    }
  }

  startLoadingProgress() {
    this.stopLoadingProgress();
    this.clearConsoleLog();
    this.loadingAnim.active = true;
    this.loadingAnim.progress = 0;
    this.loadingAnim.backendPhase = 'queued';
    this.loadingAnim.backendMessage = '';
    this.loadingAnim.runStartedAt = Date.now();
    this.setLoadingProgress(0, 'AI is optimizing routes...', 'Queueing job...', 0.22, 'idle');
    this.setLoadingSubtitle('Sending request to the backend solver...');
    this.applyDeviceBadge();
    this.setLoadingBackend('queued', 'warn');
    this.addConsoleLog('Optimization job submitted to worker queue.');
    this.startElapsedTicker();
    this.el.loading?.classList.remove('loading--minimized');
    this.el.loadingLauncher?.classList.add('hidden');

    const phasesConfig = {
      queued: { label: 'Queueing job...', until: 15, baseSpeed: 0.08, algo: 'idle' },
      processing: { label: 'Worker picked up the job...', until: 30, baseSpeed: 0.06, algo: 'idle' },
      matrix: { label: 'Building distance matrix...', until: 50, baseSpeed: 0.05, algo: 'idle' },
      solving: { label: 'Running solvers in parallel...', until: 95, baseSpeed: 0.012, algo: 'alns' },
      done: { label: 'Optimization complete!', until: 100, baseSpeed: 0.4, algo: 'done' },
    };

    const tick = (now) => {
      if (!this.loadingAnim.active) return;

      const phaseKey = this.loadingAnim.backendPhase || 'queued';
      const config = phasesConfig[phaseKey] || phasesConfig['queued'];

      const lastTick = this.loadingAnim.lastTickAt || now;
      const delta = Math.max(8, now - lastTick);

      const distance = config.until - this.loadingAnim.progress;
      let step = 0;

      if (distance > 0) {
        const factor = 0.4 + (distance / config.until) * 0.8;
        step = config.baseSpeed * factor * (delta / 16);
        this.loadingAnim.progress = Math.min(config.until, this.loadingAnim.progress + step);
      } else {
        step = 0.003 * (delta / 16);
        this.loadingAnim.progress = Math.min(99.5, this.loadingAnim.progress + step);
      }

      const currentLabel = this.loadingAnim.backendMessage || config.label;

      this.setLoadingProgress(
        this.loadingAnim.progress,
        null,
        currentLabel,
        Math.min(1, Math.max(0.1, step / 0.1)),
        config.algo
      );

      this.loadingAnim.lastTickAt = now;
      this.loadingAnim.rafId = requestAnimationFrame(tick);
    };

    this.loadingAnim.lastTickAt = performance.now();
    this.loadingAnim.rafId = requestAnimationFrame(tick);
  }

  stopLoadingProgress() {
    this.loadingAnim.active = false;
    if (this.loadingAnim.rafId) cancelAnimationFrame(this.loadingAnim.rafId);
    this.loadingAnim.rafId = 0;
    this.stopElapsedTicker();
  }

  setLoadingSubtitle(text) {
    if (this.el.loadingSubtitle && typeof text === 'string') {
      this.el.loadingSubtitle.textContent = text;
    }
  }

  setLoadingBackend(label, state = 'ok') {
    if (!this.el.loadingBackend) return;
    this.el.loadingBackend.textContent = label;
    this.el.loadingBackend.dataset.state = state;
  }

  startElapsedTicker() {
    this.stopElapsedTicker();
    if (!this.el.loadingElapsed) return;
    const start = this.loadingAnim.runStartedAt || Date.now();
    const render = () => {
      const sec = Math.max(0, (Date.now() - start) / 1000);
      this.el.loadingElapsed.textContent =
        sec < 60 ? `${sec.toFixed(1)}s` : `${Math.floor(sec / 60)}m ${Math.floor(sec % 60)}s`;
    };
    render();
    this.loadingAnim.elapsedTimer = window.setInterval(render, 200);
  }

  stopElapsedTicker() {
    if (this.loadingAnim.elapsedTimer) {
      window.clearInterval(this.loadingAnim.elapsedTimer);
      this.loadingAnim.elapsedTimer = 0;
    }
  }

  minimizeLoading() {
    if (this.el.loading?.classList.contains('hidden')) return;
    this.el.loading?.classList.add('loading--minimized');
    this.el.loadingLauncher?.classList.remove('hidden');
    this.setStatus('Optimization is running in background. Click the floating truck to reopen progress.', 'ok');
  }

  restoreLoading() {
    if (this.el.loading?.classList.contains('hidden')) return;
    this.el.loading?.classList.remove('loading--minimized');
    this.el.loadingLauncher?.classList.add('hidden');
  }

  cancelLoading() {
    this.runSession.cancelled = true;
    this.runSession.abortController?.abort();
    this.runSession.abortController = null;
    this.hideLoadingImmediate();
    this.setStatus('Optimization canceled by user.', 'error');
    this.toast('Canceled', 'Optimization was canceled.', 'error');
  }

  async completeLoading() {
    const start = this.loadingAnim.progress;
    this.stopLoadingProgress();
    const duration = 420;
    const t0 = performance.now();

    await new Promise((resolve) => {
      const step = (now) => {
        const t = Math.min(1, (now - t0) / duration);
        const eased = 1 - Math.pow(1 - t, 3);
        const value = start + (100 - start) * eased;
        const velocity = Math.max(0.22, 1 - t * 0.8);
        this.setLoadingProgress(value, 'Optimization Complete', 'Rendering maps and KPI...', velocity, 'done');
        if (t < 1) {
          requestAnimationFrame(step);
          return;
        }
        resolve();
      };
      requestAnimationFrame(step);
    });

    if (this.el.loadingCard) {
      this.el.loadingCard.classList.add('loading-card-shake');
      await new Promise((resolve) => setTimeout(resolve, 320));
      this.el.loadingCard.classList.remove('loading-card-shake');
    }

    await new Promise((resolve) => setTimeout(resolve, 180));
    this.el.loading?.classList.add('hidden');
  }

  hideLoadingImmediate() {
    this.stopLoadingProgress();
    this.setLoadingProgress(0, 'AI is optimizing routes...', 'Collecting route data...', 0, 'idle');
    this.el.loading?.classList.remove('loading--minimized');
    this.el.loadingLauncher?.classList.add('hidden');
    this.el.loading?.classList.add('hidden');
  }
}
