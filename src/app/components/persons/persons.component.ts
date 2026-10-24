import { Component, input, output } from '@angular/core';

import { FaceService, type PersonGroupInfo } from '../../services/face.service';
import { ImageElementService } from '../../services/image-element.service';
import { FilePathService } from '../views/file-path.service';

import type { PersonRecord } from '../../../../node/face/face-types';

/**
 * Person list view (design doc §8.2): a grid of person cards with the
 * representative face, an editable name, video/face counts and -- when
 * the attributes feature is on -- the per-person aggregates (§5.8).
 *
 * Phase 3 adds the manual corrections: merging people (multi-select),
 * splitting / moving (video, group)s between people or back to
 * unassigned, and pinning a person's representative face. Merges,
 * splits and moves are destructive, so each runs through a two-step
 * confirm (first click arms the action, second click executes). The
 * edits themselves are pins in the face index (see node/face), so they
 * survive re-clustering.
 */
@Component({
  standalone: false,
  selector: 'app-persons',
  templateUrl: './persons.component.html',
  styleUrls: ['./persons.component.scss']
})
export class PersonsComponent {

  readonly folderPath = input<string>();
  readonly hubName = input<string>();
  readonly darkMode = input<boolean>();
  readonly attributesEnabled = input<boolean>(false);

  readonly personClicked = output<string>();
  readonly scanRequested = output<void>();

  editingPersonId = '';
  editingName = '';

  // --- Phase 3: merge mode ---
  mergeMode = false;
  mergeSelection = new Set<string>();
  mergeArmed = false;

  // --- Phase 3: per-person group editing ---
  expandedPersonId = '';
  groupSelection = new Set<string>();
  moveTarget = '';              // personId | '__new__' (split) | '__none__' (unassigned)
  groupActionArmed = false;

  // --- Phase 3: unassigned groups ---
  unassignedMoveTarget = '';

  constructor(
    public faceService: FaceService,
    public filePathService: FilePathService,
    public imageElementService: ImageElementService
  ) { }

  /** Visible persons first (most videos first), hidden ones last. */
  persons(): PersonRecord[] {
    return [...this.faceService.persons].sort((a, b) => {
      if (a.hidden !== b.hidden) {
        return a.hidden ? 1 : -1;
      }
      return (b.videoHashes.length - a.videoHashes.length) || (b.faceCount - a.faceCount);
    });
  }

  cropPath(person: PersonRecord): string {
    return this.filePathService.createFaceCropPath(this.folderPath(), this.hubName(), person.representativeFaceId);
  }

  groupCropPath(group: PersonGroupInfo): string {
    return this.filePathService.createFaceCropPath(this.folderPath(), this.hubName(), group.representativeFaceId);
  }

  personLabel(person: PersonRecord): string {
    return person.name || ('Person ' + this.faceService.personNumber(person));
  }

  startEdit(person: PersonRecord): void {
    this.editingPersonId = person.personId;
    this.editingName = person.name;
  }

  commitEdit(person: PersonRecord): void {
    if (this.editingPersonId !== person.personId) {
      return;
    }
    this.editingPersonId = '';
    if (this.editingName.trim() !== person.name) {
      this.faceService.renamePerson(person.personId, this.editingName);
    }
  }

  toggleHidden(person: PersonRecord): void {
    this.faceService.setPersonHidden(person.personId, !person.hidden);
  }

  recluster(): void {
    this.faceService.requestRecluster(this.imageElementService.imageElements);
  }

  recalcAttributes(): void {
    this.faceService.requestAttributes(this.imageElementService.imageElements);
  }

  /** "♂ · 30s · 93%" -- sex symbol, age band, majority agreement (§5.8). */
  attributeLabel(person: PersonRecord): string {
    if (!person.attributes) {
      return '';
    }
    const attr = person.attributes;
    const symbol = attr.sex === 'M' ? '♂' : '♀';
    const band = Math.floor(attr.ageMedian / 10) * 10;
    return symbol + ' · ' + band + 's · ' + Math.round(attr.sexAgreement * 100) + '%';
  }

  // ------------------------------------------------------------------
  // Phase 3: merging
  // ------------------------------------------------------------------

  toggleMergeMode(): void {
    this.mergeMode = !this.mergeMode;
    this.mergeSelection.clear();
    this.mergeArmed = false;
  }

  toggleMergeSelection(personId: string): void {
    if (this.mergeSelection.has(personId)) {
      this.mergeSelection.delete(personId);
    } else {
      this.mergeSelection.add(personId);
    }
    this.mergeArmed = false;
  }

  /** First click arms the confirm, second click executes the merge. */
  requestMerge(): void {
    if (this.mergeSelection.size < 2) {
      return;
    }
    if (!this.mergeArmed) {
      this.mergeArmed = true;
      return;
    }
    this.faceService.mergePersons([...this.mergeSelection]);
    this.mergeMode = false;
    this.mergeSelection.clear();
    this.mergeArmed = false;
  }

  // ------------------------------------------------------------------
  // Phase 3: group-level editing (split / move / representative)
  // ------------------------------------------------------------------

  toggleGroups(person: PersonRecord): void {
    this.expandedPersonId = this.expandedPersonId === person.personId ? '' : person.personId;
    this.groupSelection.clear();
    this.moveTarget = '';
    this.groupActionArmed = false;
  }

  expandedPerson(): PersonRecord {
    return this.faceService.personById(this.expandedPersonId);
  }

  groupsOf(person: PersonRecord): PersonGroupInfo[] {
    return this.faceService.groupsOfPerson(person.personId);
  }

  unassignedGroups(): PersonGroupInfo[] {
    return this.faceService.groupsOfPerson('');
  }

  toggleGroupSelection(key: string): void {
    if (this.groupSelection.has(key)) {
      this.groupSelection.delete(key);
    } else {
      this.groupSelection.add(key);
    }
    this.groupActionArmed = false;
  }

  /** A split must move a proper subset of the person's groups. */
  canSplitSelected(person: PersonRecord): boolean {
    const total = this.groupsOf(person).length;
    return this.groupSelection.size > 0 && this.groupSelection.size < total;
  }

  groupActionPossible(person: PersonRecord): boolean {
    if (this.groupSelection.size === 0 || this.moveTarget === '') {
      return false;
    }
    return this.moveTarget !== '__new__' || this.canSplitSelected(person);
  }

  /** First click arms the confirm, second click executes split/move. */
  requestGroupAction(person: PersonRecord): void {
    if (!this.groupActionPossible(person)) {
      return;
    }
    if (!this.groupActionArmed) {
      this.groupActionArmed = true;
      return;
    }
    const keys = [...this.groupSelection];
    if (this.moveTarget === '__new__') {
      this.faceService.splitPerson(person.personId, keys);
    } else {
      this.faceService.moveGroups(keys, this.moveTarget === '__none__' ? null : this.moveTarget);
    }
    this.groupSelection.clear();
    this.moveTarget = '';
    this.groupActionArmed = false;
  }

  /** Other people the selected groups can be moved to. */
  moveTargetPersons(person: PersonRecord): PersonRecord[] {
    return this.faceService.persons.filter((p) => p.personId !== person.personId);
  }

  isRepresentativeGroup(person: PersonRecord, group: PersonGroupInfo): boolean {
    return person.representativeFaceId === group.representativeFaceId;
  }

  setRepresentative(person: PersonRecord, group: PersonGroupInfo): void {
    this.faceService.setPersonRepresentative(person.personId, group.representativeFaceId);
  }

  /** Move one unassigned group to the chosen person (reversible, no confirm). */
  moveUnassignedGroup(group: PersonGroupInfo): void {
    if (this.unassignedMoveTarget === '') {
      return;
    }
    this.faceService.moveGroups([group.key], this.unassignedMoveTarget);
  }

  formatTime(seconds: number): string {
    const total = Math.max(0, Math.floor(seconds));
    const m = Math.floor(total / 60);
    const s = total % 60;
    return m + ':' + String(s).padStart(2, '0');
  }

}
