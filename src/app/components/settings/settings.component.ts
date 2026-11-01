import { Component, Input, input, output, viewChild } from '@angular/core';

import { TranslateService } from '@ngx-translate/core';

import { ModalService } from './../modal/modal.service';

import { SettingsMetaGroup, SettingsMetaGroupLabels } from '../../common/settings-buttons';

import type { OnChanges, SimpleChanges } from '@angular/core';
import type { OnInit } from '@angular/core';
import type { SettingsButtonsType } from '../../common/settings-buttons';
import type { FaceSettings, PhotoSettings } from '../../../../interfaces/final-object.interface';
import type { FaceModelStatus, FaceStats } from '../../../../node/face/face-types';
import type { FaceDownloadProgress } from '../../services/face.service';

@Component({
  standalone: false,
  selector: 'app-settings',
  templateUrl: './settings.component.html',
  styleUrls: [
    '../buttons.scss',
    '../settings.scss',
    '../search-input.scss',
    './settings.component.scss'
  ]
})
export class SettingsComponent implements OnInit, OnChanges {

  readonly changeLanguage = output<string>();
  readonly checkForNewVersion = output<any>();
  readonly chooseDefaultVideoPlayer = output<any>();
  readonly decreaseZoomLevel = output<any>();
  readonly goDownloadNewVersion = output<any>();
  readonly increaseZoomLevel = output<any>();
  readonly openOnlineHelp = output<any>();
  readonly resetZoomLevel = output<any>();
  readonly scrollSettingsToTop = output<void>();
  readonly toggleButton = output<string>();
  readonly toggleHideButton = output<string>();
  readonly faceSettingsChanged = output<void>();
  readonly faceDownloadRequested = output<void>();
  readonly faceDownloadCancelRequested = output<void>();
  readonly photoSettingsChanged = output<void>();
  readonly faceScanRequested = output<void>();
  readonly faceReclusterRequested = output<void>();

  @Input() appState;
  readonly demo = input();
  readonly faceModelMismatch = input<boolean>();
  readonly faceModelStatus = input<FaceModelStatus>();
  readonly faceSettings = input<FaceSettings>();
  readonly faceStats = input<FaceStats>();
  readonly faceDownloadProgress = input<FaceDownloadProgress>();
  readonly faceDownloadAvailable = input<boolean>(false);
  readonly photoSettings = input<PhotoSettings>();
  readonly latestVersionAvailable = input();
  readonly settingTabToShow = input();
  @Input() settingsButtons: SettingsButtonsType;
  readonly versionNumber = input();

  readonly settingsModal = viewChild('settingsModal');

  additionalInput = '';
  editAdditional = false;
  settingsMetaGroup = SettingsMetaGroup;
  settingsMetaGroupLabels = SettingsMetaGroupLabels;

  constructor(
    private modalService: ModalService,
    private translate: TranslateService
  ) {}

  ngOnInit(): void {
    this.additionalInput = this.appState.addtionalExtensions;
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes.settingTabToShow) {
      this.scrollSettingsToTop.emit();
    }
  }

  toggleFaceEnabled(event: Event) {
    const settings = this.faceSettings();
    if (settings) {
      settings.enabled = (event.target as HTMLInputElement).checked;
      this.faceSettingsChanged.emit();
    }
  }

  /** Approximate download size shown on the download button */
  downloadSizeLabel(): string {
    const pack = this.faceSettings()?.modelPack;
    if (pack === 'quality') {
      return '~289 MB';
    }
    if (pack === 'light') {
      return '~15 MB';
    }
    if (pack === 'lvface') {
      return '~77 MB + detector';
    }
    return '';
  }

  downloadInProgress(): boolean {
    const progress = this.faceDownloadProgress();
    return !!progress && (progress.stage === 'downloading' || progress.stage === 'extracting');
  }

  downloadPercent(): number {
    const progress = this.faceDownloadProgress();
    if (!progress || !progress.total) {
      return 0;
    }
    return Math.min(100, Math.floor((progress.received / progress.total) * 100));
  }

  downloadProgressText(): string {
    const progress = this.faceDownloadProgress();
    if (!progress) {
      return '';
    }
    const receivedMb = (progress.received / 1048576).toFixed(1);
    if (progress.total) {
      return `${receivedMb} / ${(progress.total / 1048576).toFixed(1)} MB`;
    }
    return `${receivedMb} MB`;
  }

  togglePhotoEnabled(event: Event) {
    const settings = this.photoSettings();
    if (settings) {
      settings.enabled = (event.target as HTMLInputElement).checked;
      this.photoSettingsChanged.emit();
    }
  }

  toggleFaceAutoScan(event: Event) {
    const settings = this.faceSettings();
    if (settings) {
      settings.autoScanNew = (event.target as HTMLInputElement).checked;
      this.faceSettingsChanged.emit();
    }
  }

  toggleFaceAttributes(event: Event) {
    const settings = this.faceSettings();
    if (settings) {
      settings.attributesEnabled = (event.target as HTMLInputElement).checked;
      this.faceSettingsChanged.emit();
    }
  }

  changeFaceThreshold(event: Event) {
    const settings = this.faceSettings();
    if (settings) {
      const value = parseFloat((event.target as HTMLInputElement).value);
      if (!isNaN(value)) {
        settings.clusterThreshold = Math.min(0.9, Math.max(0.1, Math.round(value * 100) / 100));
        this.faceSettingsChanged.emit();
      }
    }
  }

  changeFaceModelPack(event: Event) {
    const settings = this.faceSettings();
    if (settings) {
      settings.modelPack = (event.target as HTMLSelectElement).value as FaceSettings['modelPack'];
      this.faceSettingsChanged.emit();
    }
  }

  changeFaceProvider(event: Event) {
    const settings = this.faceSettings();
    if (settings) {
      settings.provider = (event.target as HTMLSelectElement).value as FaceSettings['provider'];
      this.faceSettingsChanged.emit();
    }
  }

  editAdditionalExtensions() {
    this.editAdditional = !this.editAdditional;
    this.additionalInput = this.appState.addtionalExtensions;
  }

  applyAdditionalExtensions() {
    if (this.isAdditionalInputValid(this.additionalInput)) {
      this.appState.addtionalExtensions = this.additionalInput;
      (window as any).myElectron.sendToMain('update-additional-extensions', this.additionalInput);
      this.editAdditional = false;
    } else {
      this.modalService.openSnackbar(this.translate.instant('SETTINGS.extensionsInputError'));
    }
  }

  private isAdditionalInputValid(input: string): boolean {
    let valid = true;
    input.split(',').forEach(element => {
      if (/[^A-Za-z0-9]/.test(element.trim())) {
        valid = false;
      }
    });

    return valid;
  }

}
