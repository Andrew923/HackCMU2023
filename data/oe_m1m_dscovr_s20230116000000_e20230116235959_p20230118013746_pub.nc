CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230116000000_e20230116235959_p20230118013746_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-18T01:37:46.023Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-16T00:00:00.000Z   time_coverage_end         2023-01-16T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill         �   records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx[~�  �          @��H���@��\@Q�A�33CE���@+�@��B{Cz�                                    Bx[~f  T          @����@k�@��RB\)C�����?�ff@�ffBF�C)                                    Bx[~   T          A   ����@o\)@�33BG�C�f����?�p�@\BE�
C �=                                    Bx[~.�  �          Az���(�@G
=@��B-G�C����(�?p��@�(�BJ��C)�
                                    Bx[~=X  �          @��R��{@+�@��HB3\)C����{?�@�BL�\C-s3                                    Bx[~K�  �          @��
���?�@���B8C ����녾�\)@���BDG�C7^�                                    Bx[~Z�  �          @�33���?�=q@���BD�\C#������.{@��BJ=qC<c�                                    Bx[~iJ  T          @��H�s�
@;�@��BJ��Cp��s�
?5@�Q�Bm  C)p�                                    Bx[~w�  �          @�\�Z=q@Fff@��\BD�\C	�R�Z=q?��@\Bn33C"�                                    Bx[~��  �          @�G��1G�@���@��B=�B��3�1G�?���@�(�Bx=qC�                                    Bx[~�<  �          @��H���@�ff@��B0ffB����@(Q�@�p�Bt
=CL�                                    Bx[~��  �          @�z��dz�@8��@�z�B@�C�R�dz�?n{@�=qBf�C%ff                                    Bx[~��  �          @�\)�fff@{@�G�BS��C���fff>��R@���BpQ�C/�                                    Bx[~�.  �          @أ��K�@7�@��BJ��C	��K�?^�R@���Bs\)C$�)                                    Bx[~��  �          @�33�.{@�R@�p�BB(�C	�H�.{?aG�@���Bl33C!�q                                    Bx[~�z  �          @��\�'�@��@�  BD�RC.�'�?:�H@�G�BlQ�C$O\                                    Bx[~�   �          @���p�@�@�G�B^=qCQ��p�?
=q@�=qB�ffC'�                                    Bx[~��  u          @�{��R@(��@�=qBY=qC33��R?O\)@�{B��qC�R                                    Bx[
l  T          @�\)��@+�@��\BW  B�B���?n{@�
=B�C^�                                    Bx[  3          @�����
=@=p�@x��B?
=B�33��
=?�
=@�p�B{�CaH                                    Bx['�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[6^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[E              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[S�  �          @Ӆ�W�@�G�@UA���B�33�W�@G
=@�ffB6��C	T{                                    Bx[bP  �          @��\�1G�@n�R@$z�A�\)B�\)�1G�@'
=@l(�B0\)C��                                    Bx[p�  �          @ȣ����@L(�@>�RA㙚C�����?�(�@y��B  CY�                                    Bx[�  T          @�p����@|��@O\)A��
Cs3���@%@���B'C                                    Bx[�B  �          @�  �{�@fff@b�\B=qC	z��{�@
�H@��B5�
C�                                    Bx[��  �          @�33�Fff@@  @��HB;�C�R�Fff?��R@��HBg��C=q                                    Bx[��  T          @�(��fff@�(�@Y��A��HB���fff@K�@�G�B3�\C
��                                    Bx[�4  �          @�p��7�@��R@   A��
B�\�7�@~�R@e�B��B���                                    Bx[��  �          @�����=q@���@@  A��
C����=q@U@��B�
C�)                                    Bx[׀  "          @�33���@�Q�?�G�Axz�C����@w�@Q�A�(�C	�                                    Bx[�&  �          @ָR����@��?�=qAY��CW
����@u@E�Aڏ\C޸                                    Bx[��  T          @޸R��@��?c�
@�33C�=��@�ff@$z�A�\)Cu�                                    Bx[�r  �          @�����  @�(�?��@���C G���  @�33@=qA��
C33                                    Bx[�  �          @ٙ��|��@�Q�W
=��  B�33�|��@�  ?�Q�Af�RB��)                                    Bx[� �  
�          @�{�r�\@���p���p�B��\�r�\@�p�?aG�@�=qB�u�                                    Bx[�/d  T          @��
���\@��R@fffA���C�q���\@L��@��B#
=C��                                    Bx[�>
  �          Az����@�
=@c33A�G�C�R���@l��@��
B��C^�                                    Bx[�L�  �          @������@�
=@K�A�\)C�����@e�@�{BG�C��                                    Bx[�[V  �          @�=q���R@�33@2�\A�p�C&f���R@g
=@���B{C�                                    Bx[�i�  �          @�����Q�@���@:=qA�  C�\��Q�@p��@�ffB�RC=q                                    Bx[�x�  �          @��R����@��@:=qA��C�����@w�@��B�HC�q                                    Bx[��H  u          @�����@���@%A�Cff����@w�@���BG�C                                      Bx[���  �          @�{���R@�{@G
=A���C33���R@u�@�{B�
C�f                                    Bx[���            @�����R@��
@'�A�33C�3���R@���@��
B�\C�3                                    Bx[��:  e          @���  @�ff@��A�z�C G���  @���@�{B��C�                                    Bx[���  T          @�Q����H@��?�(�As�CQ����H@��@p  A��C��                                    Bx[�І  
�          @�\)��ff@��?�(�A\(�Cn��ff@�Q�@\��A�Q�C�)                                    Bx[��,  �          @���`��@�{?�z�AQ�B�(��`��@�@L(�A�B�Ǯ                                    Bx[���  �          @޸R�o\)@�?�=qA0��B����o\)@���@J�HAٙ�B���                                    Bx[��x  �          @���  @�
=@"�\A���C���  @tz�@�=qB��C��                                    Bx[�  
�          @����(�@�Q�@'�A�{CE��(�@u�@���B�C޸                                    Bx[��  3          @�p���z�@���@mp�A��RC
����z�@#33@�33B*  C33                                    Bx[�(j  T          @��R��{@���@�Q�B��C�=��{@%�@�ffB-
=C��                                    Bx[�7  T          @�������@�@�G�BL�C�����=�@�ffBaC2n                                    Bx[�E�  �          AQ���  @�ff@�ffB�HC���  @Q�@��HB:
=C�=                                    Bx[�T\  T          @�Q�����@��
@0  A�
=C &f����@y��@��B33C�H                                    Bx[�c  T          @�33��Q�J=q@��B�.CU���Q��'
=@�
=Bp  Cu�\                                    Bx[�q�  �          @�ff�0�׿�G�@�G�B��HCuk��0���<(�@qG�BM33C�g�                                    Bx[��N  
�          @�����ÿ���@u�Bs��C^�q�����)��@HQ�B5�HCoc�                                    Bx[���  
�          @ʏ\?���z�@��B@�C���?����R@@  A�C��                                     Bx[���  
�          @˅?����  @��B0��C�~�?����
=@)��A�Q�C�                                    Bx[��@  �          @�{�z��q�@�Q�BF��C����z����@?\)A�33C��)                                    Bx[���  
�          @��ÿ����P  @�(�BU��C{W
�������@Q�B
�C�N                                    Bx[�Ɍ  �          @�Q�xQ��X��@��
BX=qC~��xQ���=q@]p�Bp�C�W
                                    Bx[��2  
�          @�33�#33��H@�=qBg  C_�\�#33��z�@�ffB)�
Cnc�                                    Bx[���  !          @��\��(��7�@�Q�B[�Co���(���G�@b�\B�RCx33                                    Bx[��~  �          @��R�����33@�(�Bp��Cg�׿����xQ�@��\B.\)Ct�{                                    Bx[�$  �          @�  ���*=q@�
=Bv��Ck�����@�  B1�RCx
=                                    Bx[��  �          @�Q������@��B���Ci��������Q�@�33BA33Cx:�                                    Bx[�!p  �          @�ff��z��{@�B��Cm� ��z���
=@��\BH�\C{�                                     Bx[�0  �          @�p��C33>B�\@�G�Bo��C0���C33����@���B_G�COO\                                    Bx[�>�  "          @�=q�~{��z�@�p�BL(�C8.�~{��33@���B7��CM�{                                    Bx[�Mb  
�          @�ff�^{>�p�@��RBj{C.  �^{���@�Q�B]�\CK��                                    Bx[�\  T          @��
���
?O\)@�  BWz�C(����
����@�{BTz�CCff                                    Bx[�j�  D          @У�����?���@���B8\)C"޸���þ�z�@��\BAz�C7�{                                    Bx[�yT  2          @�{�b�\��G�@���B>{CC���b�\��@^{BCTǮ                                    Bx[���  
�          @�ff����@��@��BLG�CiǮ������@W�B
z�Cs.                                    Bx[���  �          @��5�mp�@�p�BW=qC��
�5���R@j=qB�\C��                                    Bx[��F  2          @Ϯ�C33�+�@�BpG�C@W
�C33�=q@��BNffCZL�                                    Bx[���  �          @����I���ff@�=qBT��CU�3�I���i��@�33B!z�Ce&f                                    Bx[�  �          @��Ϳ�z��Dz�@�(�BY��Cq����z���  @g
=B�RCy��                                    Bx[��8  T          @�Q�?8Q����@@��Bp�C�z�?8Q���G�?�{Ac�C��q                                    Bx[���  �          @����\�0��@tz�BV�C��
��\�vff@-p�BffC�1�                                    Bx[��  D          @_\)�k�����@H��B��Cg
=�k��@%BBz�CvY�                                    Bx[��*  �          @,(��E��h��@��B~(�Ce��E����?�(�B=p�Ctٚ                                    Bx[��  �          @%��E��z�H@�RBs=qCg�׿E���33?��B1��Cu�                                    Bx[�v  
�          @o\)��
=��
=@H��Bm�Cf����
=��@   B-�Cs�)                                    Bx[�)  T          @O\)�\(����R@6ffBx33CkJ=�\(��ff@�\B5�Cw��                                    Bx[�7�  "          @Y����(��p��@8��BkffCT���(���=q@�B7��Cg33                                    Bx[�Fh  �          @�  �G���p�@]p�B^��CSaH�G���@8Q�B-z�Cd�\                                    Bx[�U  �          @����
=?���?���B#�RC	�=��
=?�?�z�BN�CY�                                    Bx[�c�  �          @vff��{��Q�@\(�B�Q�Cb𤿎{�\)@7�BC�Cs�\                                    Bx[�rZ  �          @~�R�^�R���\@i��B�aHCk���^�R�Q�@C33BI33Cy��                                    Bx[��   �          @��׿!G��\(�@w�B��Ci�H�!G���
@XQ�Bf{C|�f                                    Bx[���  �          @��
��ff�s33@|(�B�(�Ct�)��ff�
�H@[�Bd��C�/\                                    Bx[��L  �          @�녿.{��(�@~{B��=CuY��.{�*�H@Q�BH{C�=                                    Bx[���  �          @�z�.{����@vffBw33Cz�f�.{�E�@AG�B/�C��q                                    Bx[���  �          @���B�\�
=@UB`�\Cz\�B�\�C�
@\)B��C�                                      Bx[��>  T          @�녿��\���@h��Bm�Cq�H���\�=p�@5B(�HCz�                                    Bx[���  �          @�\)����33@g�Bk�C[�q����R@>�RB3�\Cl�                                    Bx[��  �          @��Ϳ�p���
=@^�RBe{CdxR��p��,��@1G�B'�RCq33                                    Bx[��0  �          @�(����H�
=q@L��BK33Ck�ÿ��H�Dz�@B
ffCt��                                    Bx[��  �          @p  ����G�@&ffB1G�Cn�)����>�R?޸RA�\)Cu\                                    Bx[�|  �          @�=q���H��@C�
BG�Ck�=���H�?\)@�RBffCs��                                    Bx[�""  �          @]p��(���\)@��B1��C5�\�(��Q�@��B$CF�f                                    Bx[�0�  
�          @l(���R>��@,��B=C0����R�5@'
=B5z�CD\                                    Bx[�?n  �          @�Q��
=�u@"�\B<G�C5\)�
=�Y��@��B.\)CG�                                    Bx[�N  T          @��
���
�aG�@w�B{�CN@ ���
��
@X��BM  Ce�                                    Bx[�\�  �          @��\���;�  @x��B��qC;�H���Ϳ��H@g
=Bc�CZO\                                    Bx[�k`  �          @�����>\)@vffBtz�C00�����=q@l(�Be{CN��                                    Bx[�z  �          @�  �1녽u@8Q�B8  C50��1녿u@.{B*�HCG�                                    Bx[���  �          @�����>���@a�Bj=qC+J=��ÿL��@]p�BbffCH�                                     Bx[��R  �          @�\)�Q�>�G�@^{B\33C)� �Q�+�@[�BX��CC��                                    Bx[���  "          @��
��?+�@W
=BZ��C#���녾�
=@Y��B^�C>aH                                    Bx[���  T          @�G���?W
=@H��BM{C �����8Q�@O\)BV�HC8p�                                    Bx[��D  T          @�(��Q�?\(�@O\)BP  C 0��Q�L��@UBY�HC8�3                                    Bx[���  �          @�(��1�?s33@8��B2  C!=q�1�    @B�\B>
=C4�                                    Bx[���  �          @�(��:=q?z�H@HQ�B6=qC!ff�:=q�#�
@QG�BA�C4�
                                    Bx[��6  �          @����L��?�z�@9��B!��C !H�L��>L��@G�B0�C0xR                                    Bx[���  �          @����g
=?�{?�Q�A��
C�{�g
=?�z�@�HB�HC"(�                                    Bx[��  T          @�ff�qG�@�?�A��C�H�qG�?��@��A�Q�C h�                                    Bx[�(  �          @���xQ�@�@ ��AîCQ��xQ�?�33@$z�A�\)C 33                                    Bx[�)�  �          @�{����@	��?���A�{Cٚ����?�p�@�HA�p�C��                                    Bx[�8t  �          @�ff����@*=q?aG�A&{C}q����@��?�z�A�Q�C�f                                    Bx[�G  �          @�z���@;�?��@�ffC�R��@'�?�
=A�33C�3                                    Bx[�U�  �          @�����z�@A�>�Q�@�G�C����z�@1G�?��\Ae��C5�                                    Bx[�df  T          @�33�b�\@5?&ffA{C8R�b�\@ ��?��RA��C��                                    Bx[�s  �          @�ff�Vff@J�H?c�
A.{C�
�Vff@0  ?�A�C��                                    Bx[���  �          @��\�Vff@C33>�Q�@�33C	�H�Vff@2�\?��
A���C!H                                    Bx[��X  �          @���g
=@@  >��@�  C:��g
=@.{?�\)A���C�                                    Bx[���  T          @�{�y��@Fff?��A@(�C���y��@(Q�?���A�ffC                                      Bx[���  �          @��R��=q@o\)�\)��C���=q@fff?�  A!p�C&f                                    Bx[��J  "          @�p��}p�@P��=�\)?Q�C���}p�@E�?��A>�\C{                                    Bx[���  "          @����w�@;�>��H@�Q�C�)�w�@)��?���A���C��                                    Bx[�ٖ  
�          @���w
=@:�H?G�A��C���w
=@#33?�\)A��\C�\                                    Bx[��<  T          @�z��w�@U>B�\@
=C8R�w�@HQ�?�Q�AV�RC�                                    Bx[���  T          @�(�����@A�?W
=A�HC!H����@(��?��HA�Q�C޸                                    Bx[��  
�          @�����33@XQ�>.{?��C33��33@J�H?�
=AEC�                                    Bx[�.  T          @����  @p  >.{?�Q�C
�)��  @aG�?��
AQG�CW
                                    Bx[�"�  "          @�
=��33@|��>�=q@,��C���33@l(�?�Q�Aj{C
�                                    Bx[�1z  �          @�{�w�@�p�<��
>8Q�C�H�w�@~{?�G�AMG�CB�                                    Bx[�@   
�          @�
=�|��@����\)�0��C޸�|��@|(�?�z�A;�
C�                                    Bx[�N�  �          @�����=q@tz�#�
��Q�C
����=q@i��?���A0z�C�                                    Bx[�]l  �          @�����@n{�L�Ϳ�Q�C  ��@g
=?k�A{C�
                                    Bx[�l  �          @�ff��@w���Q�^�RC	B���@n{?��A,(�C
^�                                    Bx[�z�  �          @��R��{@w�<�>���C	Q���{@k�?�A=�C
�3                                    Bx[��^  �          @�ff�g�@_\)?&ff@�C��g�@I��?�33A�(�C                                    Bx[��  �          @���,(�@b�\?��A�p�B�u��,(�@7
=@0��BQ�CL�                                    Bx[���  �          @�G��6ff@g�@�A��\C @ �6ff@7
=@@��BC�                                    Bx[��P  �          @���'
=@^{?���A�=qB��q�'
=@:�H@�
A�z�C��                                    Bx[���  �          @���=p�@^{?L��A
=C�=�=p�@E�?��
A��HC��                                    Bx[�Ҝ  �          @�\)�=p�@`  ?���AW33C:��=p�@AG�@z�AиRCaH                                    Bx[��B  �          @�  �<��@P��?�(�A��\C5��<��@'
=@%BG�C
�                                     Bx[���  T          @~�R���@3�
?�{A�\)CxR���@�\@�B33CG�                                    Bx[���  �          @z�H��
=@8��?��
A��HB��{��
=@  @"�\B"�\C��                                    Bx[�4  "          @g
=���@9��?��RA��B����@@G�B��B��q                                    Bx[��  T          @e���@5?n{A|Q�B��)��@(�?��HA�=qB��=                                    Bx[�*�  �          @z=q�%@4z�    =uC���%@,��?Q�AEC�\                                    Bx[�9&  
�          @s33�G�?�{?c�
A\��C5��G�?\?��A��\C�q                                    Bx[�G�  �          @q��@��?�{?��A��HCG��@��?�p�?��
A�(�CǮ                                    Bx[�Vr  "          @�  �C�
@:�H>��@^{CL��C�
@.�R?��Am��C
J=                                    Bx[�e  
Z          @����&ff@Vff�#�
�{B����&ff@P��?L��A.�RC ��                                    Bx[�s�  "          @��X��@�þk��Mp�C�
�X��@ff>�@˅C5�                                    Bx[��d  
�          @o\)�Fff?��>k�@j�HC� �Fff?޸R?B�\AAG�C�                                     Bx[��
  
�          @|(��QG�@Q��G���z�C�
�QG�@�?   @�33C�=                                    Bx[���  
�          @l���QG�?�
=�k��eCǮ�QG�?�
=>��@�  C�{                                    Bx[��V  "          @z=q�c33?���>���@��HC{�c33?�33?Tz�AC�C�                                    Bx[���  �          @s33�[�?���?L��AB�\C{�[�?���?�z�A�=qC"@                                     Bx[�ˢ  �          @�\)�`  @   ?.{A�Cz��`  @p�?�{A�G�C�=                                    Bx[��H  �          @�
=�j�H@S�
=�G�?�p�C	���j�H@I��?��AAG�Cff                                    Bx[���  T          @�ff�^{@aG����R�c�
C�H�^{@]p�?0��A ��C
                                    Bx[���  
�          @���J=q@l�Ϳc�
�%�CxR�J=q@s�
>8Q�@Q�C��                                    Bx[�:  �          @��R�b�\@?uAK33C��b�\?�p�?���A��C�=                                    Bx[��  �          @���i��>���@aG�B/=qC-��i�����@_\)B-p�C=Y�                                    Bx[�#�  T          @��\�S�
���@s�
BB�
C<��S�
��G�@aG�B0Q�CL�
                                    Bx[�2,  �          @���U��!G�@uBBG�C>�3�U���@aG�B-G�CN��                                    Bx[�@�  �          @�Q��R�\��(�@��BK�C;aH�R�\���@r�\B8��CM
=                                    Bx[�Ox  
�          @�(��?\)�+�@�33BT�
C@���?\)��\@p  B<�RCR��                                    Bx[�^  T          @���333��@H��B?z�C==q�333���@8��B,Q�CL�R                                    Bx[�l�  T          @�z��Q�!G�@C�
BL��CB�f�Q쿹��@0��B3�CS^�                                    Bx[�{j  
�          @��R�`  @�\?�
=A��
C�\�`  ?У�?�Q�A�\)C�R                                    Bx[��  
�          @�����  @&ff?��
AB�\C���  @{?ٙ�A��C                                    Bx[���  
�          @�Q��dz�=���@EB#Q�C2ff�dz�@  @@  B�\C?�
                                    Bx[��\  "          @��\��(�?\)?��
A��C,:���(�=u?�\)A£�C38R                                    Bx[��  �          @�Q����
��?�G�A�G�C4s3���
��\?�Q�A���C;�                                    Bx[�Ĩ  �          @�
=�P�׾�=q@Q�B4Q�C8��P�׿�33@E�B'  CGc�                                    Bx[��N  
�          @����\�;�@N�RB+\)C;�f�\�Ϳ�=q@>�RB\)CI+�                                    Bx[���  T          @�����?��H@�
A�C n���?Y��@�A��
C(L�                                    Bx[��  �          @�  ��G�?s33@'
=B ��C&���G�>W
=@1G�B	�C1                                      Bx[��@  �          @�ff�tz�?L��@%BG�C(!H�tz�=��
@-p�BffC2ٚ                                    Bx[��  �          @�ff���\?˅?�p�A�(�C���\?���@A���C%^�                                    Bx[��  T          @�  ���R?�z�?�33A�G�Cn���R?�p�?��A��
C#��                                    Bx[�+2  �          @�\)���\?�33?��HA�{C%)���\?8Q�?��HA�Q�C*��                                    Bx[�9�  
�          @��H���H?�ff?�A���C&\)���H?�@33A��C-�                                    Bx[�H~  �          @�����G�?���?�
=A��\C$�R��G�?�R?�z�A�Q�C+L�                                    Bx[�W$  "          @�Q�����?�33?ǮA�z�C$0�����?0��?�A��
C*J=                                    Bx[�e�  T          @��H��=q?��H?��Ax  Cn��=q?��?�Q�A��\C#+�                                    Bx[�tp  �          @�ff��  ?�z�?=p�A	C���  ?��?�(�Ac�
C�3                                    Bx[��  
�          @�=q���
?�?8Q�A	p�C0����
?˅?�Ab�RC \                                    Bx[���  "          @����  ?��H?O\)A ��C
=��  ?�Q�?�p�Au��C!E                                    Bx[��b  "          @�Q����?��R?c�
A,��C!W
���?��H?�  At  C$�R                                    Bx[��  "          @��
��ff?xQ�?333Az�C(W
��ff?@  ?n{A0Q�C*�H                                    Bx[���  "          @�����R?�?��AP��C-Y����R>k�?��AeG�C1�                                    Bx[��T  �          @�=q���?333?��HAw�C*�{���>�Q�?���A�z�C/=q                                    Bx[���  
�          @�\)���
?���?s33A:�HC!�����
?�33?��A�C%0�                                    Bx[��  �          @��\�j�H@*�H>�  @L��C��j�H@ ��?n{A=C��                                    Bx[��F  "          @�ff�`��@'
=?�\@�33CW
�`��@��?�z�At��C�3                                    Bx[��  
�          @���X��@/\)?
=@���C��X��@\)?��\A�(�C�H                                    Bx[��  
�          @�z��Vff@=p�?�Q�An{C
��Vff@#�
?�z�A£�C��                                    Bx[�$8  �          @���^�R@S�
?k�A.{C}q�^�R@=p�?޸RA�ffC��                                    Bx[�2�  �          @��\�Fff@aG�?��AL(�C\)�Fff@HQ�?�A��\C�                                    Bx[�A�  
�          @���A�@c�
?�R@�Q�CaH�A�@R�\?�  A�ffC��                                    Bx[�P*  "          @�  �Z=q@N{?8Q�A\)C���Z=q@;�?�G�A�ffCO\                                    Bx[�^�  
�          @��\�\��@=p�>�p�@��RC\)�\��@1G�?�{Aa�C:�                                    Bx[�mv  "          @�{�tz�@*=q��Q쿏\)C+��tz�@&ff?z�@�p�C�\                                    Bx[�|  �          @����^{@*�H?���Au��C^��^{@�\?�=qA��\C��                                    Bx[���  �          @���g�@:=q?!G�@�\)C33�g�@*=q?��A��RC��                                    Bx[��h  �          @�  �{@w
=>��@���B�Ǯ�{@g�?�A�B�                                      Bx[��  �          @�
=�:=q@^{>W
=@*=qC���:=q@S�
?���AY�C@                                     Bx[���  �          @�(��U�@Fff?=p�A�HC	��U�@4z�?�p�A�  C�q                                    Bx[��Z  
�          @��
�C�
@Y��?z�@�  C  �C�
@I��?�33A��
C+�                                    Bx[��   
�          @����5�@Q�>��@^{C�\�5�@G�?��Ab�HCJ=                                    Bx[��  �          @����W�?�?���A��\C���W�?�?��
A��C0�                                    Bx[��L  �          @�(��\��?�(�?�=qAr�\C.�\��?�33?��
A�=qCff                                    Bx[���  �          @��R�J=q@(Q�>k�@K�CG��J=q@\)?\(�AA��C�                                    Bx[��  T          @�Q��A�@9��>�ff@�p�CJ=�A�@,��?��A|(�C
L�                                    Bx[�>  T          @��
�,��@
=q@�A�{Cff�,��?�=q@%�B
=C�R                                    Bx[�+�  �          @�33�Dz�@��?���A�z�Cs3�Dz�?�Q�@ffB�
C�=                                    Bx[�:�  T          @�ff�g�@
=?���A��C���g�?�p�?��A�Q�C��                                    Bx[�I0  �          @�=q����?�{�u�8��C u�����?�\)>��?�ffC T{                                    Bx[�W�  T          @��H�\)?�p�?   @���C�\�\)?�?uAF{C��                                    Bx[�f|  �          @��H�]p�@��?\(�A8(�CǮ�]p�@
�H?�A���C��                                    Bx[�u"  T          @�=q�W�@%�?@  A�
C���W�@�?��A��C\)                                    Bx[���  
�          @�33�Y��?�(�?���As33C�H�Y��?�z�?�G�A���C�3                                    Bx[��n  1          @���}p�@G���G�����C33�}p�@�R>�G�@�
=C��                                    Bx[��  �          @����a�?��
>k�@VffC@ �a�?�
=?&ffA�C�H                                    Bx[���  "          @��R�j=q?�ff�#�
�{C�j=q?�33�B�\�-p�C�                                    Bx[��`  "          @����l��?��������C"c��l��?�Q�z�H�\��C�                                    Bx[��  T          @�z��o\)?�{�+��{C���o\)?�(���\)�vffCJ=                                    Bx[�۬  w          @�Q��o\)?�p�����c�
C!H�o\)?��R>W
=@1�C�                                    Bx[��R  c          @�z��vff@   ��R�Cz��vff@�\)��\Cs3                                    Bx[���  
�          @���p��@{>��@��C���p��@33?xQ�ALz�Cz�                                    Bx[��  
�          @�Q��XQ�@"�\����33C{�XQ�@{?\)@�(�C�                                    Bx[�D  
i          @����hQ�?��Ϳ�z��
=C���hQ�@z�5���CO\                                    Bx[�$�  	`          @��\�qG�@ff�������C��qG�@�>#�
@
�HC�R                                    Bx[�3�            @�  �h��@녾\)��p�C���h��@ ��>��
@�ffC(�                                    Bx[�B6  1          @�G��aG�@����G����HC� �aG�@�=�?У�CY�                                    Bx[�P�  
�          @�{�aG�@
�H�W
=�7�CO\�aG�@
=q>�z�@���Ck�                                    Bx[�_�  �          @�ff��@i��?8Q�A!�B��H��@X��?��
A��B�Q�                                    Bx[�n(  
�          @�녿   @b�\?˅A�B�
=�   @E@ffBp�B�Q�                                    Bx[�|�  T          @�G���33@`  ?�
=A�RB�aH��33@>{@*�HB'  B�aH                                    Bx[��t  
�          @�  ����@Tz�?�p�A��HB�Q쿬��@:=q@�B��B��                                    Bx[��  �          @�녿�\)@Y��?���A��
B���\)@Dz�?�{A�G�B�3                                    Bx[���  �          @����33@J=q?��An�\B����33@6ff?�(�A��C ٚ                                    Bx[��f  "          @�Q���@]p�?8Q�A&ffB�W
���@Mp�?�(�A�z�B���                                    Bx[��  �          @�Q��%�@<��=�\)?�ffC+��%�@7
=?8Q�A)C
=                                    Bx[�Բ  T          @���.{@A녾\)��C޸�.{@?\)?�@�CE                                    Bx[��X  
�          @�G��\)@^{�\)��=qB�W
�\)@Z�H?!G�A\)B�8R                                    Bx[���  
�          @��\��\@U?   @�=qB�����\@I��?�(�A�33B���                                    Bx[� �  
�          @��R�G�@I����ff���C���G�@J�H>u@G�Cz�                                    Bx[�J  T          @�(��1G�@X�þL���'�C5��1G�@W
=?
=q@��HC�                                    Bx[��  T          @�{�=p�@P�׿z���Q�C8R�=p�@S�
>\)?��C��                                    Bx[�,�  �          @�  �B�\@J�H�Tz��+�
C���B�\@Q녾���C��                                    Bx[�;<            @��׿�\)@�G��5���B垸��\)@�33>8Q�@\)B�                                      Bx[�I�  
�          @�  ���@�33�G�� ��Bߞ����@�p�=�G�?���B��                                    Bx[�X�  
(          @�(���R@hQ�c�
�=�B�.��R@o\)���Ϳ��
B�3                                    Bx[�g.  "          @�G����H@����
=q����B�.���H@�=q>���@�\)B�                                    Bx[�u�  �          @�(����@u��
=q��  B����@w
=>�z�@�Q�Bۀ                                     Bx[��z  T          @�G���@XQ쿇��v�\B�.��@aG����
���B�#�                                    Bx[��   E          @���R@N{��z�����B�L���R@\�Ϳ333��B��q                                    Bx[���  �          @�p����@G
=��\)��z�B�{���@XQ�n{�N=qB���                                    Bx[��l  "          @�����@E���z���(�B�(���@S33�=p��%p�B�=q                                    Bx[��  �          @�����@HQ쿝p���=qB��q��@Tz�\)��ffB��                                    Bx[�͸  
�          @�G��=q@G��^�R�G\)B���=q@N�R�W
=�<(�B��\                                    Bx[��^  
�          @\)�p�@C33�333�"�RC �3�p�@HQ�u�Y��C =q                                    Bx[��  
�          @��H�(�@Vff�J=q�333B�W
�(�@\�ͽ�Q쿝p�B��                                    Bx[���  
�          @��R��33@mp���
=��
=B�LͿ�33@n{>�33@���B�33                                    Bx[�P  "          @�G���@k��5�G�B��f��@p  =L��?0��B�                                      Bx[��  T          @�����@g��E��&=qB�R��@l�ͼ��
�uB                                    Bx[�%�  �          @�Q��p�@n�R�����{B��
��p�@qG�>W
=@8��B�ff                                    Bx[�4B  �          @x�ÿ�@N{�h���]B�3��@U�u�k�B���                                    Bx[�B�  �          @s33��(�@[�<�?   B�33��(�@W
=?8Q�A6=qB�                                    Bx[�Q�  �          @l(��
=q@e���
=���HB��
=q@e>��R@��B���                                    Bx[�`4  �          @tzῙ��@e��G���\)B��H����@b�\?(�AQ�B�\)                                    Bx[�n�  
�          @N�R��(�@�R�aG���
=B�LͿ�(�@
=��
=�\)B�.                                    Bx[�}�  T          @g
=>W
=@e�>.{@333B�Q�>W
=@^{?aG�Ab�RB��                                    Bx[��&  
�          @z�H?�@dz�?(��AQ�B��3?�@W�?���A�=qB�aH                                    Bx[���  "          @�G�?�ff@c33?z�A  B|��?�ff@W�?�G�A�=qBw�                                    Bx[��r  T          @�=q?�z�@tz�>�(�@�(�B�.?�z�@j�H?�A�
=B���                                    Bx[��  "          @�Q�?��
@���>�
=@�{B�Ǯ?��
@w�?���A��HB�p�                                    Bx[�ƾ  w          @�z�?��@|��>\)?�33B�?��@vff?fffAIB��                                    Bx[��d  1          @�p�?��H@�z�>��@���B�8R?��H@~�R?���A{�
B��
                                    Bx[��
  
�          @���?�@�?#�
A
=qB�z�?�@~�R?�
=A���B��R                                    Bx[��            @�ff=�@��?fffAG�
B��{=�@s�
?�z�A�{B�\)                                    Bx[�V  c          @\)�k�@q�?�ffAv�RB���k�@`��?�  A���B��                                    Bx[��  T          @J�H��@;�?(�A;�B��H��@0��?�z�A�  B���                                    Bx[��  
�          @C�
� ��?��H?Y��A���C��� ��?\?�z�A�33C��                                    Bx[�-H  
�          @'
=�"�\>Ǯ>aG�@�(�C+@ �"�\>�{>�z�@�{C,Q�                                    Bx[�;�  
�          @�H��=L�ͽ#�
�s33C2�
��=L�ͼ��8Q�C2�                                    Bx[�J�  T          @p  �,(�@\)?s33Ar�RC5��,(�@G�?���A���C
=                                    Bx[�Y:  
�          @��\���@AG�?�G�A��C h����@.�R?�ffA��HC8R                                    Bx[�g�  
�          @��\�33@E�?�{A�B�ff�33@1�?�z�A��HC�
                                    Bx[�v�  
�          @vff�z�@<(�?�=qA�(�B�.�z�@)��?���A�ffB���                                    Bx[��,  �          @~{�ff@A�?�  Ai��B��{�ff@333?��A��\C�                                    Bx[���  �          @x���Q�@8��?��\As�C���Q�@*=q?��
A���C�f                                    Bx[��x  �          @������@@��?�
=A�=qC �=���@/\)?��HAȣ�C!H                                    Bx[��  �          @�{�;�@6ff?aG�AB=qC�{�;�@(��?���A�(�C	�R                                    Bx[���  T          @~{�#33@1G�?���A��C�f�#33@ ��?�
=AɮC�                                     Bx[��j  �          @mp���@!�?�(�A��RC{��@G�?�33Aՙ�C+�                                    Bx[��  T          @h���   @�R?Tz�AT��C:��   @�\?�G�A�33C	}q                                    Bx[��  �          @`�׿��R@%�?��A�z�B�LͿ��R@�
?�(�A�  C�                                     Bx[��\  �          @l�Ϳ��R@2�\?��A��HB�{���R@   ?�{A�  C ��                                    Bx[�	  �          @a녿��
@333?�
=A���B����
@#33?�z�A�ffB��q                                    Bx[��  �          @e���  @6ff?�  A��
B�8R��  @%?�p�A�B�G�                                    Bx[�&N  w          @Z=q�33@?L��A\(�C���33@
�H?���A�Q�C�                                     Bx[�4�  c          @\���1�?��H��Q쿾�RC���1�?�Q�>�  @�  C)                                    Bx[�C�  "          @a��?\)?^�R�У����C#���?\)?�녿��H��C0�                                    Bx[�R@  "          @L���   >�G�������C*��   ?=p���(���\C#k�                                    Bx[�`�  T          @녿�33?���   �Qp�C�
��33?��׾��
��\C=q                                    Bx[�o�  �          @#�
�Q�?�\��G���\)C&�{�Q�?8Q쿔z��ٮC!n                                    Bx[�~2  c          @@���Q�>B�\��  �{C/W
�Q�>��H�ٙ���C(ff                                    Bx[���  �          @/\)���?���������C$�{���?Q녿�G���C��                                    Bx[��~  
�          @$z��\?�33>���Az�C���\?Ǯ?+�A|��C
��                                    Bx[��$  �          @n{�  @8��>��@��\B��3�  @3�
?=p�A:ffC ��                                    Bx[���  �          @w
=��@6ff?�A��B�Ǯ��@ ��@�\B�
B�p�                                    Bx[��p  �          @^�R���
@&ff?�z�A�z�B�
=���
@�\@z�B�\B���                                    Bx[��  �          @S33�
=@����z�C
�
=@��#�
�8Q�C	E                                    Bx[��  �          @{��5@
=����ƣ�CT{�5@
=���\���
CW
                                    Bx[��b  T          @U���?�(��L�ͿfffCG���?���>�=q@�33C��                                    Bx[�  �          @}p��p�@(�@��B��CB��p�?�\@1G�B1�Cff                                    Bx[��  T          @�G�����@\)@1G�B.C� ����?�G�@FffBJ{CW
                                    Bx[�T  �          @^�R���\?Ǯ@,(�BT�C���\?�{@:=qBo�RC
�=                                    Bx[�-�  
�          @Dz�}p�?��
@%Bh=qB���}p�?W
=@0��B��\CxR                                    Bx[�<�  �          @��H��  ?���@XQ�Bc33C���  ?J=q@c33Bv(�C                                    Bx[�KF  �          @��H��
?�z�@Y��BPp�C��
?���@g�BdC�                                    Bx[�Y�  	�          @���
=@�@��B�C���
=@G�@0  B-
=C8R                                    Bx[�h�  T          @e����\?^�R@I��B�C�H���\>�Q�@P  B�C$\                                    Bx[�w8            @Mp��
=>L��@EB�G�C!^��
=��z�@E�B�
=CN�                                    Bx[���  c          @`  ��  �L��@Tz�B�p�C7  ��  ��@QG�B�G�CQ��                                    Bx[���  
�          @u����?���@QG�Bp  C!H���?Y��@\(�B��{CxR                                    Bx[��*  �          @i�����H@{@33BB�Q쿚�H@�@)��B?  B�\)                                    Bx[���  T          @\)�>B�\@B�\BP��C/}q���\)@A�BP=qC:�)                                    Bx[��v  
�          @�{�:=q>���@j=qBM��C.(��:=q���@j=qBM�
C9.                                    Bx[��  
(          @�=q�J�H>�p�@EB0�\C-G��J�H����@G�B2
=C5�\                                    Bx[���  
�          @���Dz�B�\@7�B+��C7xR�Dz��R@3�
B'z�C?Y�                                    Bx[��h  
�          @q��\)���H@8Q�BM�C@G��\)�h��@1G�BC��CJ0�                                    Bx[��  �          @Dz��{�fff@��BQ��CQ+���{��  @\)B>��CY�q                                    Bx[�	�  
�          @Dz�\(���=q@,��B|  CgxR�\(���(�@ ��B_Co��                                    Bx[�Z  
�          @S�
���L��@�\B4�
CH�q������@��B&CP{                                    Bx[�'   b          @_\)�{���@
=qB  CK8R�{��\)?�p�B�CP�                                    Bx[�5�  T          @j=q�>{�+�@G�BQ�C@�R�>{�u?�33A��HCE��                                    Bx[�DL  "          @g
=�/\)<��
@�\BC3�\�/\)���R@G�B��C:�                                     Bx[�R�  T          @Vff�0�׿(��?�
=A�G�CA}q�0�׿fff?���A�33CF�                                    Bx[�a�  T          @Tz��G
=���?�{A�Q�C8�=�G
=��
=?��A�33C;��                                    Bx[�p>  �          @Q��;��(�?��RA��C?���;��G�?��A���CBٚ                                    Bx[�~�  
�          @Mp��?\)���\>�ffAG�CFٚ�?\)����>���@�(�CG                                    Bx[���  �          @b�\�P  ��{>.{@/\)CJ�3�P  ��\)�����CJ�)                                    Bx[��0  "          @�(��u����H��������CH޸�u���녿���p�CG�                                    Bx[���  
�          @\(��;���Q�?��A�\)C5�=�;��u?���A��C8�                                     Bx[��|  �          @l���7�?&ff@��B{C'5��7�>�{@G�B�C-B�                                    Bx[��"  �          @i���0  ?�R@�B�HC'Q��0  >���@B!�C-�{                                    Bx[���  "          @qG��1G�?=p�@�BffC$�3�1G�>�
=@=qB"�C+p�                                    Bx[��n  �          @
�H���>�?ٙ�BQQ�C.zῧ���G�?ٙ�BQ�C8�3                                    Bx[��  "          @"�\����>.{@33B_��C,�׿��׽�G�@�
B`\)C8��                                    Bx[��  
�          @333��\)���
@  BX�HC6�f��\)�\@{BT�\CA(�                                    Bx[�`  �          @8Q���H��  ?�
=A�G�C9����H��G�?���A�\C>J=                                    Bx[�   
�          @,(���Ϳ@  ?@  A�33CE���ͿW
=?&ffA`(�CG
=                                    Bx[�.�  �          @33��Q�\(�?:�HA�z�CK���Q�s33?(�A}�CN(�                                    Bx[�=R  
Z          @����}p�?�33A���CP@ ������?�G�A�ffCS�)                                    Bx[�K�  
~          @.{��
�^�R?�Q�A�  CH����
���\?���A��CK�                                    Bx[�Z�  0          @&ff�(��fff?�ffA��HCJaH�(����
?k�A���CM5�                                    Bx[�iD  
�          @"�\���s33?�  A���CL�������?^�RA��CN��                                    Bx[�w�  x          @�R��\���?uA��CN���\��33?Q�A���CQs3                                    Bx[���  
�          @�׿�=q���?��AᙚCW�f��=q���\?c�
A�CZ                                    Bx[��6  �          @G���\��?E�A���C@����\���?333A��RCC�                                    Bx[���  
�          ?����\�0��?fffA���CP�����\�L��?L��A�33CTJ=                                    Bx[���  "          ?�{��=q�Ǯ?5A���CG�H��=q��?&ffA�\)CK�\                                    Bx[��(  T          ?��Ϳ�\)>8Q�?&ffA�z�C*����\)=���?+�A�C.ٚ                                    Bx[���  �          ?���  >��>\A{�
C�=��  >�
=>�(�A���C!�
                                    Bx[��t  x          @e��XQ�?s33?(�A��C$G��XQ�?^�R?8Q�A:�\C%�{                                    Bx[��  
�          @dz��^�R?�>�(�@���C+�\�^�R>�>��HA (�C,}q                                    Bx[���  "          @QG��L(�?
=>�A�C)z��L(�?�?�A��C*�=                                    Bx[�
f  �          @Fff�>�R?0��?   AG�C'
=�>�R?�R?z�A-G�C(B�                                    Bx[�  "          @E��:�H?.{?(��AFffC&��:�H?
=?=p�A^{C(��                                    Bx[�'�  
�          @Dz��333?}p�?.{AP(�C ��333?fff?L��As�C"+�                                    Bx[�6X  b          @E��*=q?�(�?��A#
=C
=�*=q?��?8Q�AW33CL�                                    Bx[�D�  "          @C�
���?�\?(��AH��C� ���?�
=?^�RA�=qC�                                    Bx[�S�  T          @C�
� ��@�>u@���CaH� ��@\)?�\A��C�                                    Bx[�bJ  T          @S33��@{>k�@���C�
��@�?�A�C�                                    Bx[�p�  �          @]p����R@1G�?�A  B�ff���R@,(�?Y��Ab�\B�                                      Bx[��  "          @Dz��
=@\)��G��(�B��ÿ�
=@�R>.{@Tz�B�
=                                    Bx[��<  �          @Z�H��\)@;��\)��B��q��\)@>�R�aG��k�B�                                      Bx[���  "          @L�Ϳٙ�@+���z���
=B���ٙ�@,(�<��
>���B�aH                                    Bx[���  �          @P���Q�@녿Tz��n�\C��Q�@
=���!��C
                                    Bx[��.  
L          @U�>L��?�
=�5��l(�B��R>L��?��R�'��RffB�                                    Bx[���  b          @L�Ϳ��
@z����p�B�ff���
@�\����	33B�u�                                    Bx[��z  T          @.�R�fff?��
���
���B��fff?�Q쿨�����B��                                    Bx[��   T          @�R��
=?�������HBҨ���
=?�
=��z��=qB�=q                                    Bx[���  �          @'����?�  >�z�AC�H���?��H>�
=AW�C��                                    Bx[�l  �          ?�p���ff?�?���B/ffC޸��ff>\?�33B7��C#Ǯ                                    Bx[�  �          @33��=q?E�?z�HA�=qC��=q?(��?��A���CaH                                    Bx[� �  �          @�����?s33?���B��C�����?O\)?�ffB�C�R                                    Bx[�/^  �          @{��33?�{?z�As�
C쿳33?��?@  A���C^�                                    Bx[�>  F          ?�z�z�H?�(���G��[�B����z�H?�(�=L��?�33B��=                                    Bx[�L�  b          @�Ϳ��R?Ǯ�z����C s3���R?�{����5��B�\                                    Bx[�[P  �          ?\�\)?���>\Ay��B���\)?�ff?�A���B��                                    Bx[�i�  x          ?\�#�
?�p�>��
Ae�B�  �#�
?�Q�>�G�A�  B�z�                                    Bx[�x�  b          ?��;���?�R��ff�`z�B�zᾨ��?:�H�z�H�K(�B�Q�                                    Bx[��B  
Z          @,(����H�z���H8RCe�=���H�����p�.CU�R                                    Bx[���  �          @�#�
�ff�u���HC���#�
���R�����
C��\                                    Bx[���  
�          @Q�>���3�
�����뙚C�"�>���(Q�����HC�h�                                    Bx[��4  b          @K�?(��?u���33B]��?(��?�������kz�Bt�                                    Bx[���            @[�?=p�@?\)��{��33B�?=p�@G
=�����z�B�\                                    Bx[�Ѐ  0          @b�\?J=q@N{��Q���{B�u�?J=q@Tz�\(��c33B�G�                                    Bx[��&  
�          @u�?:�H@HQ�����Q�B��\?:�H@P�׿�(���{B���                                    Bx[���  �          @�=q@(�@(��-p��
=B'��@(�@����R��B433                                    Bx[��r  "          @�p�?޸R@=p���
=����Bn33?޸R@HQ�����{Bs�
                                    Bx[�  T          @���?��@^{��(���33B��\?��@g���\)���HB���                                    Bx[��  �          @��R?���@h�ÿ�G����B�k�?���@qG���33��G�B�u�                                    Bx[�(d  "          @|��?z�H@U��ff��z�B�L�?z�H@_\)��(���Q�B��3                                    Bx[�7
  
�          @��?(�@xQ�����ӮB�
=?(�@�����=q���RB�Ǯ                                    Bx[�E�            @�=q?=p�@a녿�
=��z�B�u�?=p�@i�������(�B�33                                    Bx[�TV  �          @r�\?�(�@C�
����ffB��\?�(�@N{�\��ffB�z�                                    Bx[�b�  
�          @u�@33@�R���ffB=�@33@(����\)BH                                      Bx[�q�  
�          @`  @��?������"��A��R@��?�=q�	���ffB�H                                    Bx[��H  
�          @n�R?޸R@
�H����#\)BM33?޸R@Q��(���BWp�                                    Bx[���  
�          @\��@33?L���
=�<
=A��@33?�G����3��A��                                    Bx[���            @`  ?��?��\�#33�Ap�B�
?��?��R���5=qB��                                    Bx[��:  
�          @�G�?�?�(��Mp��XffB(�?�?�  �Dz��JB0�                                    Bx[���  
�          @w�?��?�G��7
=�R
=B��?��?�G��/\)�E�\B �                                    Bx[�Ɇ  T          @�33?333@|(��   ��(�B��
?333@~{����	��B���                                    Bx[��,  �          @��\?�z�@n�R��=q�v�HB��\?�z�@s�
�@  �*{B�=q                                    Bx[���  
Z          @z=q@ff@=p����\��Q�BZ��@ff@C�
��G��s33B^(�                                    Bx[��x  �          @�p�@(Q�@�������B@(Q�@z��p�� �
B%�H                                    Bx[�  T          @r�\@��u�1G��B(�C�5�@�    �1��C�C�                                      Bx[��  
�          @o\)@�
���H�\)�+ffC�"�@�
��  �'��7ffC�                                    Bx[�!j            @L(�?��ÿ\��p��(�
C��?��ÿ����ff�5�C��                                    Bx[�0  
�          @�
=��ff@dz�@	��A��
B��쿆ff@XQ�@(�B
G�B֔{                                    Bx[�>�  �          @�{�8Q�@{�?�p�A���BȨ��8Q�@qG�@33A�
=Bɀ                                     Bx[�M\  "          @���@z=q?�p�A��HB�\��@n�R@�\A�ffB�                                    Bx[�\  
�          @��
?Y��@�{@
�HA�\)B���?Y��@�  @   A�{B��f                                    Bx[�j�  "          @�\)?��@�Q�?���A�=qB�  ?��@��?�
=A��B�                                    Bx[�yN  
�          @��
?&ff@g
=@\)A���B���?&ff@Z�H@!�B��B��=                                    Bx[���  �          @������R@P��@&ffB�HB��þ��R@B�\@7
=B,�B��q                                    Bx[���  �          @���@*=q@7
=?��
A�33B<33@*=q@.�R?�  AŅB6�H                                    Bx[��@  T          @�33?��H@k�?�z�Ax��Bx
=?��H@e�?���A�=qBuQ�                                    Bx[���  �          @��R?���@e�?��A�  B�
=?���@[�@z�A�{B���                                   Bx[�  �          @�33?�@P��@��Bp�B���?�@C�
@)��B"  B��
                                   Bx[��2  
(          @����@N{@
=B�B�k���@A�@&ffB!�B�p�                                    Bx[���  	�          @��?0��@l��?�ffA�(�B���?0��@c�
@�A�z�B��                                    Bx[��~  �          @�
=?(��@Tz�@�B	�B��=?(��@H��@$z�B�B�B�                                   Bx[��$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�p              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�)              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�7�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�Fb              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�U              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�c�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�rT              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�"              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�0�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�?h              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�\�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�kZ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�z               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�"              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�)�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�8n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�G              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�U�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�d`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�s              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��R              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��D              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�ِ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��6              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�(              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�"�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�1t              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�N�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�]f              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�l              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�z�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��X              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��J              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�Җ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�*z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�9               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�G�  	m          @�G�@�
=��33�Q���
C�AH@�
=��{�aG��{C�j=                                    Bx[�Vl  
Z          @�\)@�p���\>�Q�@qG�C���@�p����
>�z�@E�C�|)                                    Bx[�e  �          @��R@�p���z�\�\)C�{@�p���33��G���(�C�'�                                    Bx[�s�  �          @��@��H��Q�Tz��#�C��@��H��z�aG��.{C��q                                    Bx[��^  �          @��H@�녿�{����  C�1�@�녿Ǯ��p���  C���                                    Bx[��  "          @��\@��ÿ������=qC�K�@��ÿ�=q�{���C���                                   Bx[���  �          @��R@��ÿ�Q�s33�)�C�AH@��ÿ�z῁G��4��C�p�                                   Bx[��P  �          @��H@�ff�   �ٙ���Q�C���@�ff��Q��G���C�0�                                    Bx[���  	�          @�(�@���������=qC��R@�����ff����C�Q�                                    Bx[�˜  �          @��H@�33��=q��Q����
C���@�33���
���R���HC�Ф                                    Bx[��B  �          @\��@Fff�Y����(���{C�\)@Fff�O\)��  ��{C��R                                    Bx[���  T          @�\)@�����������(�C��q@���xQ���R��C�k�                                    Bx[���  T          @��@��ÿ�{���H��{C���@��ÿ�ff� �����HC��                                    Bx[�4  "          @�@����33�
=���C��)@�����p�����33C�:�                                    Bx[��  "          @��@��H��\)�{�îC�Ff@��H��ff�G���z�C��                                    Bx[�#�  �          @�@��\����33��(�C�C�@��\���\�ff��  C���                                    Bx[�2&  
�          @��\@�33�����Q����C�8R@�33�z�H��(����C��                                    Bx[�@�  "          @��H@�p���녿��
���C��=@�p���=q������(�C�
                                    Bx[�Or  "          @��@��R�L�Ϳ�(���
=C�p�@��R�=p��޸R��33C��q                                    Bx[�^  
�          @�z�@��\�B�\������C�@��\�5�˅��C��                                    Bx[�l�  
�          @�=q@�녿s33��{�b�HC��3@�녿fff����g�C�+�                                    Bx[�{d  T          @���@����Y�������k�C�5�@����O\)��\)�p(�C�o\                                    Bx[��
  
�          @�z�@��
�녿Ǯ��z�C���@��
��������  C�
                                    Bx[���  �          @�z�@����=q��=q���RC�K�@���k���=q��p�C���                                    Bx[��V  "          @��
@��H��Q��{��z�C���@��H���R��\)��p�C�*=                                    Bx[���  
�          @�33@�(��k����H����C���@�(��B�\���H��\)C��                                    Bx[�Ģ  "          @�33@��;.{��\)�z�\C���@��;������{�C�8R                                    Bx[��H  
�          @���@��������33C�)@����Ϳ����C�h�                                    Bx[���  
�          @�@�(�    �У����<�@�(�=L�ͿУ����?!G�                                    Bx[��  "          @�(�@����L�Ϳ�Q���C���@�������ٙ���Q�C���                                    Bx[��:  
�          @���@xQ�L�Ϳ��H��=qC���@xQ�#�
���H��ffC��3                                    Bx[��  T          @��
@j=q�W
=��{��p�C�^�@j=q�#�
��\)��{C���                                    Bx[��  �          @�=q@l(�>�z�У�����@�@l(�>��ÿУ�����@��                                    Bx[�+,  T          @��@~�R=u��33����?Tz�@~�R=��Ϳ�33��\)?�\)                                    Bx[�9�  
�          @��@Vff?��
=�=qAG�@Vff?z���
G�A��                                    Bx[�Hx  �          @vff@�?B�\�1G��=�A�z�@�?Tz��0  �;�A��R                                    Bx[�W  �          @r�\@,��?c�
���Q�A�\)@,��?s33��p�A��\                                    Bx[�e�  
�          @z�H@8Q�?�{��H�ffA�ff@8Q�?�����G�A���                                    Bx[�tj  T          @\(�@(�>�
=�� �RA33@(�>������A.=q                                    Bx[��  
Z          @C33@���R��\)��G�C���@��
=������{C���                                    Bx[���  �          @P��>��
>���@  ¤33Bp�>��
>�{�@  ¢L�B7�R                                    Bx[��\  �          @�?z�?   ��33¡=qB#33?z�?(����\L�B:ff                                    Bx[��  "          @���?Q녿�\��p��)C��?Q녾Ǯ�� .C�E                                    Bx[���  �          @�  >�������Q��\C���>���G���G��C�z�                                    Bx[��N  �          @\(����#�
�P  aHCf�\�����P��k�Cc33                                    Bx[���  "          @Tz���G��J=q�qCk�ÿ��5�K��CiG�                                    Bx[��            @�����
�(Q���=q�J33Cg���
�"�\���
�NQ�Cf�f                                    Bx[��@  
�          @��\������\)�f  Cj������R�����jG�Ci                                    Bx[��  "          @�Q쿮{��H���gz�Cp�)��{�z���\)�k��Co�\                                    Bx[��  �          @�녿�\)�=q��{�h��Cl{��\)��
����m
=Cj��                                    Bx[�$2  �          @�녿�p��ff��33�wCf����p���p���z��{��Cd�                                    Bx[�2�  y          @��R���
�!G���  �d�Cj�q���
�=q����h33Ci�{                                    Bx[�A~  /          @�z��ff�%������l�
Ck#׿�ff�{���H�p�HCi�f                                    Bx[�P$  T          @�Q���R��33�����a  CX���R�Ǯ��{�c�HCW�                                    Bx[�^�  �          @�G���(��G���33�^  Ca����(���
=�����ap�C`k�                                    Bx[�mp  T          @����ff���fff�`Q�Ct8R��ff�ff�h���d��Csh�                                    Bx[�|  
�          @�����������z=qz�C^�3�����\)�{�{C\�                                    Bx[���  y          @�z��R���
�{�Cz�q��R�ٙ��~{��Cy��                                    Bx[��b  /          @�=q�W
=���
�tz�k�C�B��W
=�����vff�qC���                                    Bx[��  �          @��Ϳ�p������{�|Q�C_s3��p������
=�z�C]�q                                    Bx[���  "          @����R��\�����c
=C[aH��R��Q���{�e�CY�q                                    Bx[��T  �          @��(�����{�n=qCYu��(���=q��
=�p��CW�H                                    Bx[���  T          @�z����e��4z���Cl� ���aG��8����HCl{                                    Bx[��  �          @vff���׿������H�C`녿��׿�=q�
=�K=qC_��                                    Bx[��F  �          @w
=��@K������ffB���@Mp���(���B��3                                    Bx[���  "          @���(��@녿�{��{C+��(��@z����Q�C
��                                    Bx[��  G          @����
=@l(����� ��B�{�
=@l�ͽ��
����B�\                                    Bx[�8  
�          @��H>��H@�p��#�
��Q�B���>��H@�p�=��
?z�HB���                                    Bx[�+�  
�          @˅?G�@�=q?L��@�G�B��R?G�@���?h��Ap�B��                                    Bx[�:�  /          @��?p��@��\@\(�A��B�Ǯ?p��@���@c33A�Q�B��\                                    Bx[�I*  �          @��H�aG�@���@�=qB �
B��{�aG�@��@��B%
=B���                                    Bx[�W�  �          @�녿���@��H@x��A��B�����@ȣ�@�Q�B �
B���                                    Bx[�fv  T          @�
=���@�ff@�  B�RB��Ϳ��@��
@��BB�{                                    Bx[�u  
�          @�{�s33@�G�@��BffBĔ{�s33@θR@�p�Bp�B�Ǯ                                    Bx[���  �          @��ÿ�  @�G�@���BffBը���  @�
=@��
B
G�B�\                                    Bx[��h  �          @�33�.{@���@�Q�B�
BꞸ�.{@�{@�33BffB�L�                                    Bx[��  "          @�Q��z�@�@�
=B{B�Q��z�@��@��BB��H                                    Bx[���  y          @�
=�\)@�@u�B =qB�
=�\)@��
@z�HB�B߀                                     Bx[��Z  /          @��#33@�
=@X��A�B�
=�#33@��@_\)A���B�z�                                    Bx[��   G          @ʏ\?^�R@�33?�G�A���B���?^�R@�=q?���A�G�B��                                    Bx[�ۦ  
�          @�
=@�Q�@�R����n=qA��@�Q�@ �׿�  �hz�A���                                    Bx[��L  
�          @��
@�33?���XQ���RAM@�33?����Vff����AU                                    Bx[���  
�          @�{@��H@�R�<(���A���@��H@��:=q����A��H                                    Bx[��  "          @Ϯ@�@8���%����
A���@�@;��!���(�A�                                    Bx[�>  �          @���@��@z��W���A�\)@��@Q��U����\A�p�                                    Bx[�$�  �          @ə�@��
@Tz��#�
��B��@��
@W
=�   ��\)B�                                    Bx[�3�  "          @�ff@����G�����p�C�P�@�����
������C�~�                                    Bx[�B0  T          @�33@���?�G��{��(�A`��@���?�ff�����Q�Aip�                                    Bx[�P�  T          @�z�@����ff�>�R���C�3@�������?\)�	Q�C�b�                                    Bx[�_|  
�          @��\@+�@Mp��QG��
=BHG�@+�@P���N{�=qBJ{                                    Bx[�n"  �          @{�?�ff@B�\�������B��)?�ff@C�
��=q���HB��                                    Bx[�|�  �          @Tz�@)��?�?&ffA5�B
�@)��?�=q?+�A=G�B
{                                    Bx[��n  /          @�G��\)@��׽�\)�s33B��Ϳ\)@���    �uB���                                    Bx[��  
�          @�zῌ��@u>�  @g
=B��
����@u>��R@��RB��H                                    Bx[���  �          @�33��p�@<(�@B�B�33��p�@:=q@Q�B(�B��                                    Bx[��`  
�          @u�?Y��>��
@(Q�B�A��
?Y��>�\)@(��B���A�ff                                    Bx[��  �          @�33@p  ?���?��A���A��@p  ?�?�z�A��A���                                    Bx[�Ԭ  "          @�p�@<(�@H��@K�B�
B;=q@<(�@E@N�RBffB9p�                                    Bx[��R  �          @�p�@xQ�?��@S33B��A�Q�@xQ�?޸R@U�B  A�\)                                    Bx[���  �          @�(����R@g���ff��ffB��쿾�R@h�ÿ��R��  B���                                    Bx[� �  �          @��H��@��\�E����B����@��
�@����p�B�p�                                    Bx[�D  T          @�G���33@�(��J=q�33B�3��33@�p��E��{B�L�                                    Bx[��  "          @�G��H��@���dz��ffB�k��H��@���`����B��R                                    Bx[�,�  
�          @�=q�_\)@�  �`  ��HC^��_\)@����[���z�C�                                    Bx[�;6  �          @��!�@1G����Xp�Cp��!�@5��z��U�C��                                    Bx[�I�  T          @��
�,(�?�{���R�lz�CL��,(�?�Q����j�\C(�                                    Bx[�X�  �          @���P  @	�����N��C�\�P  @{�����L�C�                                    Bx[�g(  "          @�p��c�
@�p��B�\�ᙚC �f�c�
@��R�>�R��ffC ��                                    Bx[�u�  �          @ָR�r�\@}p��vff�33C��r�\@����r�\�	��C^�                                    Bx[��t  
�          @�=q�n�R@�Q��?\)����C���n�R@����;��ӮCO\                                    Bx[��  �          @�{��Q�@�\)�o\)��RC����Q�@����k�� G�C(�                                    Bx[���  �          @�p�����@q��u��\C
�f����@u�r�\�\)C
:�                                    Bx[��f  T          @�����@�
=�=q��z�C����@�Q�����C�R                                    Bx[��  
�          @�(���H@�(�����vffB���H@�z�\�Dz�B�                                      Bx[�Ͳ  T          @�\)�j�H@�
=��z��;�B�p��j�H@����=q�0Q�B�B�                                    Bx[��X  �          @�\��(�@�33��Q��=��C:���(�@��
��\)�3�C�                                    Bx[���  �          @�33�Q�@��R�#�
���
B���Q�@�  �{��{B�\)                                    Bx[���  �          @�R�E@�ff���x  B�{�E@�
=��=q�l(�B��H                                    Bx[�J  
�          @��s�
@��
�Q���33B��s�
@�(��:�H��z�B��                                    Bx[��  .          @�\��z�@��=L��>�p�B�����z�@��>\)?��B���                                    Bx[�%�  
�          @���Q�@�{?
=@�G�B�k���Q�@�?.{@�{B��                                     Bx[�4<  T          @�����Q�@��?.{@�B��f��Q�@�\)?E�@��HB���                                    Bx[�B�  �          @�\���@���?s33@�ffB�����@�(�?��@�(�B�=q                                    Bx[�Q�  
�          @��\�dz�@��?W
=@�ffBꙚ�dz�@���?p��@�B�                                    Bx[�`.  z          @����a�@�G�?�R@�\)B�L��a�@���?8Q�@�
=B�\)                                    Bx[�n�  
4          @�ff���@�
=?��\@�33B������@�ff?�{A ��B�{                                    Bx[�}z  �          @��r�\@���?�A&=qB�\)�r�\@�Q�?\A1�B�                                    Bx[��   
�          @�\)�J�H@�33?���A?
=B���J�H@ڏ\?ٙ�AK33B���                                    Bx[���  �          @����,(�@�G�?�
=A�HB�Ǯ�,(�@���?��A\)B��)                                    Bx[��l  �          @��H�)��@��?}p�@�B�L��)��@�Q�?��AQ�B�aH                                    Bx[��  
�          @���9��@��?��A%��B���9��@�  ?��RA2{B�\                                    Bx[�Ƹ  
�          @�G��=p�@�\)@�A��\B� �=p�@�{@!�A��RB�R                                    Bx[��^  H          @���AG�@�Q�@8��A�\)B��)�AG�@θR@>�RA��B�(�                                    Bx[��  .          @���@��@[�Aޏ\B����@�Q�@aG�A��HB�k�                                    Bx[��  
�          A녿��H@�
=@�p�BA�B�����H@�(�@��BD�HBڞ�                                    Bx[�P  T          A ���A�@�z�@��B�HB� �A�@�=q@�z�B�HB��                                    Bx[��  z          A녿��@��R@��BL�Bߊ=���@��
@�(�BO�
B�\)                                    Bx[��  `          A���  @�z�@�{B`ffBͮ��  @���@�Q�Bc�RB�B�                                    Bx[�-B  
�          A �׿�\@X��@�RB�\B�#׿�\@R�\@�Q�B�BŮ                                    Bx[�;�  T          A{��@�=q@���Bl�B��Ϳ�@~{@޸RBp\)B�8R                                    Bx[�J�  
�          A��#�
@aG�@�z�B���B�ff�#�
@Z=q@�{B��RB�k�                                    Bx[�Y4  T          A33=�\)@|��@��
Bt
=B���=�\)@u@�Bw�\B��                                    Bx[�g�  �          A  ?�@z=q@�Bt�RB��\?�@s33@�Bx33B��                                    Bx[�v�  T          A��>�z�@j=q@�B}�
B�
=>�z�@c33@��B��B�                                    Bx[��&  �          A�?J=q@K�@��B��
B���?J=q@Dz�@��B��\B��                                    Bx[���  �          A�H?}p�@��@���Bbz�B�Q�?}p�@�  @�33Be��B��3                                    Bx[��r  �          A�?��@N�R@��B�� B��?��@G�@�=qB�.B���                                    Bx[��  �          A (�?�@$z�@�\B��fB�33?�@��@��
B��=B�.                                    Bx[���  T          @��
>Ǯ@5@�RB���B�=q>Ǯ@.�R@�  B�k�B��{                                    Bx[��d  T          @�  �c�
@y��@ƸRBd�Bͣ׿c�
@s33@ȣ�Bhz�B�G�                                    Bx[��
  �          @�Q����@�33@��\B8��B��f���@�Q�@��B<G�B�R                                    Bx[��  "          @�  ��=q@�=q@���BR�B��
��=q@�
=@�
=BV{B���                                    Bx[��V  �          @�\)�.{@s33@��Bo��B�8R�.{@l(�@�
=BsG�B���                                    Bx[��  `          @�녾��@+�@�G�B�33B��=���@$z�@�\B�\B��
                                    Bx[��  T          @�����
@,��@ӅB���B�zὣ�
@%@���B�� B���                                    Bx[�&H  �          @�=q����@0��@أ�B�Q�B�uþ���@)��@��B�8RB���                                    Bx[�4�  �          @陚�k�?��R@�Q�B��B�L;k�?�\)@�G�B�  B��                                    Bx[�C�  
�          @�33>�@#33@��HB�
=B�aH>�@(�@�(�B���B��                                    Bx[�R:  
�          @�=u@>{@���B��B��{=u@6ff@ڏ\B���B�z�                                    Bx[�`�  �          @�>k�@�
@�
=B�{B��\>k�@(�@�Q�B�B��                                    Bx[�o�  T          @��>\)@��@ᙚB�aHB�Ǯ>\)@��@�\B�W
B�aH                                    Bx[�~,  �          @�z�?�\@ ��@�p�B�aHB�
=?�\@��@޸RB�Q�B��                                    Bx[���  
�          @�\>���@�R@��
B���B���>���@
=@�p�B��B��H                                    Bx[��x  �          @��>#�
@p�@�{B���B��>#�
@@�\)B���B�u�                                    Bx[��  �          @���>u@p�@��B�\B��f>u@@ۅB��B�Q�                                    Bx[���  "          @�=q?#�
?ٙ�@�G�B�\B���?#�
?�=q@�=qB���B��R                                    Bx[��j  �          @�33?�
=?G�@�B��B��?�
=?(��@�{B�.A��                                    Bx[��  "          @�p�?z�H?��@ָRB�8RBV�?z�H?��H@׮B��HBK�H                                    Bx[��  �          @ۅ?L��?�z�@���B��)B��R?L��?��@��B���B�                                    Bx[��\  �          @�z�?���?���@�(�B�B�BK33?���?��R@���B��HBA=q                                    Bx[�  �          @�\)?��?�(�@�p�B��BU��?��?�\)@�{B���BM�                                    Bx[��  
�          @�p�?�\)?G�@���B��A��\?�\)?.{@��B��3A�=q                                    Bx[�N  �          @�z�?�\)?�Q�@���B���BG�?�\)?��@���B�8RBQ�                                    Bx[�-�  �          @�
=@-p�?Q�@�{Bp�\A��@-p�?8Q�@�ffBqAm                                    Bx[�<�  �          @�Q�@��?��@��B{�
A�33@��?p��@�(�B}�A���                                    Bx[�K@  
f          @�\)@(��?p��@��Bq  A�ff@(��?W
=@�p�Brp�A�p�                                    Bx[�Y�  
�          @��@�{>u@�ffB433@QG�@�{>#�
@��RB4p�@��                                    Bx[�h�  �          @�
=?}p�?��
@�(�B�.Bt  ?}p�?�@��B�(�Bm�\                                    Bx[�w2  "          @����@4z�@���Bs�B��ÿ��@-p�@��RBw��B��
                                    Bx[���  
�          @�33��=q@*�H@��\Bq�RB��f��=q@$z�@�(�Bv  B�z�                                    Bx[��~  
�          @\���@�@��Bx�
B�녿��@��@���B|�RB�k�                                    Bx[��$  
�          @�33�.{?�@�G�B��B��{�.{?�Q�@��\B�u�B�Q�                                    Bx[���  :          @���>��?�@�B�Q�B���>��?�ff@��RB���B���                                    Bx[��p  
�          @ȣ�@:�H>��@��Bw��@�z�@:�H>��@�  Bx�@7�                                    Bx[��  "          @˅����?\@��B��=Bʀ ����?�33@�{B��B�z�                                    Bx[�ݼ  �          @��
�z�H@Q�@�G�B�#�B��Ϳz�H@  @��HB�ffB�
=                                    Bx[��b  �          @�ff��?˅@�\)B�aHB�=q��?�Q�@�Q�B��)B�u�                                    Bx[��  "          @�녾���?�Q�@�ffB�#�B�8R����?�@��B��{B��)                                    Bx[�	�  
Z          @�33��R@;�@�p�BZ33B�aH��R@4z�@��B^33C L�                                    Bx[�T  "          @�{�p�@'
=@��Ba�HCQ��p�@\)@�
=Be�RC�)                                    Bx[�&�  �          @˅�z�@J=q@�Q�BO��B��\�z�@C33@��\BS��B��\                                    Bx[�5�  
Z          @�p��(��@?\)@���BN  Cs3�(��@8Q�@��HBQ�HC��                                    Bx[�DF  T          @���U@p��@��
B#�HC���U@j=q@�ffB'Cp�                                    Bx[�R�  �          @�33�^�R@k�@�\)B�Ch��^�R@e�@��B#\)C33                                    Bx[�a�  �          @љ��^�R@p��@�=qBC�\�^�R@j=q@��B�C��                                    Bx[�p8  T          @����\��@dz�@���B
=C
=�\��@^�R@��B�HC�{                                    Bx[�~�  	�          @���e�@j�H@uB=qCY��e�@dz�@z�HB(�C{                                    Bx[���  �          @�\)�i��@p  @s33B{CG��i��@i��@x��B  C                                      Bx[��*  �          @�{�e�@\(�@�33B33C33�e�@U@�B"{C	\                                    Bx[���  �          @�33�tz�@Y��@�33B�C
ff�tz�@R�\@�B��CE                                    Bx[��v  �          @ٙ�����@r�\@i��B\)C
ff����@l��@p  B�C
                                    Bx[��  �          @�Q���G�@���@>�RA�p�C
z���G�@|(�@E�A�
=C                                    Bx[���  
�          @�
=��\)@z=q@"�\A��HC
����\)@u@(��A��\Cc�                                    Bx[��h  T          @����p�@xQ�@�\A�ffCG���p�@u�@��A��C��                                    Bx[��  �          @�ff���H@q�?�ffA�
=C� ���H@n�R?�33A��\C�)                                    Bx[��  
f          @������\@\(�@/\)Aٙ�C����\@W
=@5�A�\)C�=                                    Bx[�Z  T          @�33���\@HQ�@Q�A�Q�C(����\@C�
@p�A�p�C��                                    Bx[�    �          @��R�s33@AG�?�=qA�p�C���s33@>{?�z�A���C�q                                    Bx[�.�  `          @�{��\)@z�@P��B>=qC �H��\)@�R@Tz�BC
=C��                                    Bx[�=L  
�          @�{���@�\@,��B!��C�)���@p�@0��B&�C�\                                    Bx[�K�  "          @���z�@(Q�@G�B�RC =q�z�@$z�@ffB�C �                                    Bx[�Z�  
�          @��33@�@�B�CxR�33@
=@\)BG�CJ=                                    Bx[�i>  
�          @����@  @
=q?��HA�p�C@ �@  @
=@G�A���C��                                    Bx[�w�  �          @���!G�?��R@AG�B7�RCc��!G�?�33@C�
B:��C޸                                    Bx[���  "          @����4z�?�z�@B�\B+��Cp��4z�?���@EB.�C�=                                    Bx[��0  �          @��\�\��@.{?�Q�A��HC�R�\��@*�H@G�A��CO\                                    Bx[���  
�          @���g
=@C33?�ffAw
=CǮ�g
=@@��?��A�(�C&f                                    Bx[��|  �          @�{�e�@6ff?��AM�C}q�e�@3�
?�\)A]�C�\                                    Bx[��"  �          @��
�g�@AG�?�z�A\z�C��g�@?\)?�  An{Cp�                                    Bx[���  T          @�p��R�\@`��?xQ�A7
=C33�R�\@^�R?�=qAK33Cu�                                    Bx[��n  
�          @�(��R�\@[�?��AE�C޸�R�\@X��?��AYG�C&f                                    Bx[��  
�          @�  �Q�@O\)?�{AX��Cff�Q�@L��?��HAl��C�R                                    Bx[���  "          @��
�J�H@H��?�
=Al��CO\�J�H@Fff?��
A�Q�C��                                    Bx[�
`  "          @����QG�@5?��\A���C
�R�QG�@333?�{A�{Cc�                                    Bx[�  �          @����\��@%?O\)A)��C#��\��@#�
?c�
A;
=Cn                                    Bx[�'�  �          @��\�]p�@4z�?�G�AL  C�
�]p�@2�\?���A^�\C.                                    Bx[�6R  
�          @�  �Y��@/\)?��\AQp�C��Y��@-p�?���Ad  Cs3                                    Bx[�D�  �          @�\)�_\)@.{?333A�HC��_\)@,��?G�A!G�CG�                                    Bx[�S�  "          @���\��@C�
?�ffAL(�C
k��\��@A�?�33A`Q�C
                                    Bx[�bD  �          @����dz�@A�?aG�A*�RC�R�dz�@?\)?z�HA>ffC                                    Bx[�p�  "          @�  �fff@A�?
=@�p�C���fff@@��?0��A�RC.                                    Bx[��  �          @�ff�c33@C33?�\@�  CW
�c33@A�?�R@��C�                                    Bx[��6  �          @�\)�fff@@  >��H@��C5��fff@>�R?
=@�  Cc�                                    Bx[���  "          @�33�W�@B�\?!G�@��C	���W�@AG�?:�HA
=C
.                                    Bx[���  �          @���R�\@Fff?\(�A+�C�3�R�\@Dz�?xQ�AAC	                                      Bx[��(  
�          @��R�\@Mp�?Tz�A#
=C���R�\@K�?p��A9C�R                                    Bx[���  
�          @��Q�@N{?fffA1p�Cz��Q�@L(�?��\AHQ�CǮ                                    Bx[��t  "          @���Fff@Mp�?�ffAS\)C�Fff@J�H?�z�Ak33CaH                                    Bx[��  
�          @���@��@C33?�  A��C�H�@��@@  ?�{A�p�C{                                    Bx[���  
�          @�Q��K�@@  ?��
ATQ�C���K�@=p�?��Ak�C	\                                    Bx[�f  
Z          @����a�@Mp�>�{@��C	�q�a�@L��>�@�G�C	޸                                    Bx[�  �          @�z��e@Tzἣ�
���C	:��e@Tz�=���?��HC	=q                                    Bx[� �  �          @�
=�aG�@_\)����@��C:��aG�@`  ����p�C+�                                    Bx[�/X  T          @����`  @U�B�\��HCW
�`  @W��!G����C�                                    Bx[�=�  �          @�z��e@N�R�L���\)C
��e@P�׿.{��
=C	Ǯ                                    Bx[�L�  �          @�z��Z�H@AG��W
=�&�\C
���Z�H@C33�8Q���HC
J=                                    Bx[�[J  �          @����X��@9�����\�M�CxR�X��@<(��fff�5��C�                                    Bx[�i�  "          @����N�R@A녿aG��4��C�)�N�R@C�
�B�\��C��                                    Bx[�x�  �          @��H�E@7��}p��RffC	(��E@:=q�^�R�9G�C�=                                    Bx[��<  "          @�33�G
=@:=q�z�����C���G
=@;�����C�3                                    Bx[���  �          @��N{@)��>#�
@\)C�{�N{@(��>�=q@n�RC��                                    Bx[���  
Z          @mp��4z�@�þL���G�C�R�4z�@�����Ϳ˅C��                                    Bx[��.  "          @���Tz�@*=q��
=��(�C=q�Tz�@+����R��(�C
                                    Bx[���  
�          @����e�@?\)��\)�W33C(��e�@A녿}p��>ffC�q                                    Bx[��z  �          @��n{@C33�xQ��5C�3�n{@E��W
=���CW
                                    Bx[��   
�          @���mp�@N�R��33�R{C
���mp�@Q녿�G��8(�C
��                                    Bx[���  T          @���j�H@E��ff�F{C��j�H@HQ�k��,Q�C�=                                    Bx[��l  "          @�ff�^�R@8�ÿ��\�{\)CE�^�R@<�Ϳ�33�a��C                                    Bx[�  "          @����O\)@=q���H���CaH�O\)@�R������G�C�\                                    Bx[��  {          @����Z=q@#�
��=q��=qC!H�Z=q@(Q쿺�H���
Ch�                                    Bx[�(^  
�          @�ff�QG�@#�
��{��33C�R�QG�@(Q쿾�R��ffC8R                                    Bx[�7  T          @�(��X��@�Ϳ������C&f�X��@ �׿����~�\C�                                    Bx[�E�  �          @�z��e�@ff�}p��Pz�C���e�@���aG��9G�C5�                                    Bx[�TP  �          @�  �_\)@��   ��ffC�_\)@�þǮ��ffC��                                    Bx[�b�  �          @�\)�@��@'���ff��{C
���@��@*�H�����C
\)                                    Bx[�q�  �          @Mp���@�\��=q���C!H��@��z�H����Cs3                                    Bx[��B  �          @;�����?�
=���H�ĸRCff����?�p���\)����C�{                                    Bx[���  T          @\)��?��ÿ\)�O�
C�\��?˅���333C)                                    Bx[���  �          @ff���
?�ff�
=����C�ÿ��
?��ÿ��j=qC�                                    Bx[��4  �          ?��H��p�?�\)���
=C�q��p�?�33����c�
C&f                                    Bx[���  �          ?��H����?W
=���
�-G�C�R����?Y����\)�ffC��                                    Bx[�ɀ  �          ?��ÿ���?E������@��C�f����?J=q��z��)C(�                                    Bx[��&  T          ?�����\?(��
=q����CxR���\?#�
��\���HC��                                    Bx[���  �          ?�ff��  ?E���R����C�=��  ?L�Ϳz����C�H                                    Bx[��r  �          ?�  ��  ?(������ffC���  ?#�
����  C�                                    Bx[�  �          ?�\)�xQ�?.{�5���RC�{�xQ�?8Q�+����HCs3                                    Bx[��  �          @�\��\)?�=q�#�
����Cٚ��\)?����G��1�C��                                    Bx[�!d  �          @�\��(�?�{�8Q�����C�)��(�?�\)���B�\Cp�                                    Bx[�0
  �          @
=q��?�p���\�]��C����?�  ��ff�@��C!H                                    Bx[�>�  �          ?�\���?�G������Q�C녿��?��
��33�4��CaH                                    Bx[�MV  �          ?�녿�ff?z�(����C  ��ff?(����ɮC�3                                    Bx[�[�  �          ?��ÿ��H>W
=����;�HC)����H>�zῦff�9\)C&��                                    Bx[�j�  �          ?E���z�=�G��.{  Cuþ�z�>���.{aHCaH                                    Bx[�yH  �          @p  �z����8���K
=C?p��zᾞ�R�:�H�M(�C;�
                                    Bx[���  �          @�=q�/\)�\�S33�G��C;�H�/\)�L���Tz��IG�C8{                                    Bx[���  �          @�\)�H�þ�33�aG��@(�C:G��H�þ���a��A(�C6�3                                    Bx[��:  �          @��R�S33���
�l���@�HC5h��S33>��l(��@��C1Ǯ                                    Bx[���  �          @���E��������RG�C;���E��8Q����\�S�C7W
                                    Bx[�  �          @�  �I����\)���Sp�C9(��I���#�
��{�T(�C4�=                                    Bx[��,  �          @�33�G���Q����H�Y
=C5�3�G�>#�
���H�X�HC1{                                    Bx[���  �          @��
�C33>�33�G
=�533C-� �C33?��E��3G�C*+�                                    Bx[��x  �          @�(��E�>�Q��\(��?�C-h��E�?\)�Z=q�=��C)��                                    Bx[��  �          @�\)�j=q?^�R�J=q� (�C&���j=q?���Fff��C#��                                    Bx[��  �          @�Q�����?aG��7
=�
=C'�f����?�ff�333��
C%\)                                    Bx[�j  �          @��H�qG�?c�
�8Q��C&� �qG�?���4z��G�C$G�                                    Bx[�)  �          @��H�w�?���G���\C$��w�?�p��C33���C"Y�                                    Bx[�7�  T          @�  �e?W
=�@����RC&�\�e?��\�<���(�C$�                                    Bx[�F\  �          @�33�aG�?��R�Fff���C �H�aG�?�
=�AG����C�                                    Bx[�U  �          @�G��[�?z�H�5���C#���[�?�z��0���=qC!Q�                                    Bx[�c�  �          @���q�?�z��'��  C�=�q�?����!���p�Ch�                                    Bx[�rN  �          @��q�?�p��{��33C!��q�?��������C�{                                    Bx[���  �          @�������?��R�33�ڸRC#�
����?���{��{C"\                                    Bx[���  �          @��\�i��?��
�G����CB��i��?��
�H����CaH                                    Bx[��@  �          @��
�c�
@   �ff�ٙ�C�q�c�
@Q��(��ʣ�C#�                                    Bx[���  �          @�p��QG�@33������Cٚ�QG�@�H��p���Q�C}q                                    Bx[���  �          @�ff�B�\@ff�޸R����CE�B�\@p���=q��(�C                                      Bx[��2  �          @�=q�3�
@   ����C
^��3�
@%��G���G�C	^�                                    Bx[���  �          @p���p�@'����
���\C8R�p�@+��\(��T  C�=                                    Bx[��~  �          @k��A�?���(���  C�f�A�?\��\)��z�Cn                                    Bx[��$  �          @�Q��|��?!G��L(���\C*���|��?W
=�H����C'�3                                    Bx[��  T          @���j�H?!G��tz��6�C*=q�j�H?c�
�qG��3�RC&W
                                    Bx[�p  T          @��
�p  ?(��p  �2��C*� �p  ?^�R�l���/��C&�3                                    Bx[�"  �          @������\>B�\�l(��(p�C1T{���\>�ff�j�H�'33C-�                                     Bx[�0�  �          @����w���p���  �>G�C9��w���\)�����?33C5)                                    Bx[�?b  �          @��\�x�þ�����?
=C:���x�þ�����\�@G�C6:�                                    Bx[�N  �          @�  �w
=�   ��\)�=p�C;s3�w
=�L����  �>�C7                                      Bx[�\�  �          @�p��|(�����G��5��C;��|(��B�\��=q�7ffC6�\                                    Bx[�kT  �          @��
�\)����z=q�1{C:��\)�B�\�|(��2p�C6��                                    Bx[�y�  �          @�G���Q�\)�{��*��C5�{��Q�>���{��*C1�R                                    Bx[���  �          @�
=�qG�?���mp��-��C$@ �qG�?�=q�hQ��(�HC �                                    Bx[��F  �          @��
�ff@J=q�e�)p�B�#��ff@Z�H�Vff���B��                                    Bx[���  �          @����E@G��fff�-  C���E@!��[��"�C�3                                    Bx[���  �          @�\)�4z�?�ff�`  �A�CT{�4z�?Ǯ�X���9�C!H                                    Bx[��8  �          @���#33?c�
�L���Gp�C ���#33?����HQ��A
=C\                                    Bx[���  T          @�Q����?��\�j�H�Z�RCٚ���?�ff�e��S{C}q                                    Bx[���  �          @�z��5?޸R�7
=�"�Cz��5?����.{�{Cp�                                    Bx[��*  T          @�33�@  @��&ff�
G�C��@  @{�=q���\C�                                     Bx[���  �          @�G��333@���@���33C��333@'
=�3�
�G�C�                                    Bx[�v  �          @�\)�33@��U��3�Cs3�33@+��HQ��&Q�C��                                    Bx[�  �          @��H�#33@(��S�
�,�RCB��#33@,(��Fff��HCk�                                    Bx[�)�  �          @��
�.{@�
�S�
�+�C��.{@$z��G
=���C��                                    Bx[�8h  �          @���#�
@
=q�J�H�-�RC�{�#�
@=q�>�R�!z�C�                                     Bx[�G  �          @��H�J=q?�=q� �����C���J=q?�\�Q����C�R                                    Bx[�U�  �          @�\)�L(�?Ǯ����C��L(�?޸R�	�����RC^�                                    Bx[�dZ  �          @�Q��I��?�
=��
�C�R�I��?�{�
�H��33Cn                                    Bx[�s   �          @���J�H?�Q��p���Cu��J�H@���\��G�C5�                                    Bx[���  �          @�  �P  @
=�-p���HC��P  @��!��(�Cc�                                    Bx[��L  �          @�Q��(Q�?�(��*=q��C#��(Q�@(��\)�
=C@                                     Bx[���  �          @�
=�*�H?��-p�� (�C�{�*�H@��#33��HC�3                                    Bx[���  �          @�33�4z�?��
�333�   C���4z�@ ���(���(�Cu�                                    Bx[��>  �          @��R�Dz�?��,(����Cs3�Dz�@��!G��	�\CxR                                    Bx[���  �          @�=q�?\)?�p��%��C  �?\)?�Q���H�z�C�                                    Bx[�ي  �          @�G��W
=?���������C�\�W
=@�{��\C
=                                    Bx[��0  �          @�G��U�?�������\C���U�@����RC�3                                    Bx[���  �          @����O\)?���#33��HC�f�O\)@ff�Q����RC�                                    Bx[�|  �          @�Q���=q?8Q��ff���C)���=q?n{�G����C'.                                    Bx[�"  �          @��R���?��
�)���  C%�q���?�G��"�\����C"�R                                    Bx[�"�  �          @�  ���\?�ff�,(��=qC%�����\?���%����C"}q                                    Bx[�1n  T          @�ff����?�33�*=q��RC#�q����?���"�\����C �3                                    Bx[�@  T          @�(��w
=?�\)�*=q�=qC xR�w
=?����!����Cn                                    Bx[�N�  T          @��H�?\)?��H�N{�(�CǮ�?\)@\)�A���C)                                    Bx[�]`  �          @�G��AG�?�=q�O\)�*\)C�\�AG�@��C�
��
C�3                                    Bx[�l  T          @�{�G�?��U�*�
Ck��G�@���J=q�p�C�                                     Bx[�z�  �          @��R�R�\?�ff�O\)�#
=CQ��R�\@ff�C�
�33C�=                                    Bx[��R  �          @����G�?�33�L(��(G�C.�G�?�Q��AG����C+�                                    Bx[���  �          @�\)�C�
?Ǯ�N�R�,�C��C�
?�{�C�
�"(�CǮ                                    Bx[���  �          @���=p�?У��C�
�(��C��=p�?�z��8�����C
                                    Bx[��D  �          @�z��<(�?�  �8Q��$�\C  �<(�?�G��.{���C                                      Bx[���  �          @�Q��C33?xQ��1��#��C"h��C33?�p��*�H�\)C�                                    Bx[�Ґ  �          @\)�*=q?�{�(Q��%�C��*=q?�{�\)��HC�)                                    Bx[��6  �          @q��!�?�=q����"ffCO\�!�?Ǯ��
�Q�CY�                                    Bx[���  �          @dz����?��R��H�+�C0����?�(�����{C@                                     Bx[���  �          @mp��ff?�\)�!G��+z�C���ff?�{�Q��\)C}q                                    Bx[�(  �          @k���?k��&ff�4{CW
��?��\)�+�CL�                                    Bx[��  �          @j�H�#33?=p��!G��-��C#���#33?}p����&G�C�{                                    Bx[�*t  �          @y���%�?���*�H�-CG��%�?�33�#33�#��C�
                                    Bx[�9  �          @����/\)?��H�,���(
=C
�/\)?�p��#�
���C�)                                    Bx[�G�  �          @���0  ?�{�:=q�-�HC���0  ?�33�0���"�RC\                                    Bx[�Vf  �          @\(��	��?����+  C��	��?����(���HC޸                                    Bx[�e  �          @`���$z�?O\)�{��C"��$z�?��
�Q��(�C
                                    Bx[�s�  �          @g
=�1�?.{�����C&E�1�?fff�
=���C"                                      Bx[��X  �          @\)�?\)?��\�=q���C!0��?\)?�G���\��C{                                    Bx[���  �          @����5?���(����C���5@�
�(��(�C�                                    Bx[���  �          @�z��#33?�(��/\)�&�
C  �#33@   �#33���C��                                    Bx[��J  �          @�p��$z�?�  �1G��&��C��$z�@�\�$z��\)C�{                                    Bx[���  �          @����$z�?��H�4z��$�\C�3�$z�@  �&ff�
=C
�{                                    Bx[�˖  T          @���'�@��B�\�*�CB��'�@�333�Q�C
33                                    Bx[��<  �          @���C�
@
=�L(��"��CxR�C�
@(��<����HCu�                                    Bx[���  �          @�z��H��@G��J�H�!(�C:��H��@ff�;���C(�                                    Bx[���  �          @�
=�Fff?����C33� ��C��Fff@
�H�5��G�C�                                    Bx[�.  �          @���?\)?�ff�C33�$�HC���?\)@��5�
=C��                                    Bx[��  �          @��H�7�?�ff�.{��HCٚ�7�@�   ���C�f                                    Bx[�#z  �          @�G��0��?��.{���C�3�0��@
=� ���33C�                                    Bx[�2   �          @����-p�?���0���!z�C���-p�@�#33��\Cz�                                    Bx[�@�  �          @����)��?�=q�5�%Cn�)��@���'��Q�C(�                                    Bx[�Ol  �          @�Q��,(�?�\�2�\�#C�f�,(�@z��$z���C\)                                    Bx[�^  �          @�{�#33?�  �5��)�\CxR�#33@�
�'
=��C�                                    Bx[�l�  �          @���!G�?�G��333�){C��!G�@z��%��=qC��                                    Bx[�{^  T          @���(�?���2�\�*�\C���(�@ff�#�
�{C:�                                    Bx[��  �          @��H�Q�?���3�
�-33C��Q�@ff�%��\)C
�                                    Bx[���  �          @����G�?��2�\�/�\CQ��G�@Q��#�
�
=C��                                    Bx[��P  �          @��
�ff?���4z��,��C8R�ff@���%��  C�\                                    Bx[���  �          @���ff?�p��:�H�.33C��ff@�
�*�H�  C�=                                    Bx[�Ĝ  �          @�\)�z�@ ���:=q�.  C\�z�@�)���ffC�R                                    Bx[��B  �          @�����@��;��.��C	O\���@=q�*=q�Q�C�                                    Bx[���  �          @�=q���@(��=p��,C����@!��*�H��C�
                                    Bx[���  �          @�ff��@  �E�/�HCh���@'
=�333�C8R                                    Bx[��4  �          @����!G�?ٙ��\)�p�C��!G�?��R�G��  C�f                                    Bx[��  �          @s33��H?�z�������C����H?�Q��
�H�  Cc�                                    Bx[��  �          @n�R�(�?�G��ff��C@ �(�?������ffC��                                    Bx[�+&  �          @|(��%�?���   �\)Cp��%�?�Q����=qC�                                    Bx[�9�  �          @{����?�p��2�\�6(�C�{���@�
�!G�� �
C5�                                    Bx[�Hr  �          @{�����@�
�-p��.�RC}q����@Q����G�CaH                                    Bx[�W  �          @~�R���?���;��>�\C�)���@\)�*�H�)Q�C�                                    Bx[�e�  �          @����ff?�
=�5��2�C	ff�ff@G��$z���C�                                    Bx[�td  �          @qG��	��?�33�(��!z�C
���	��@(����z�C�                                     Bx[��
  �          @mp�����?���!��,�C�����@(��G��{C�R                                    Bx[���  �          @���� ��@�
�3�
�1\)CJ=� ��@���!G����C��                                    Bx[��V  �          @|�Ϳ��
@   �9���<�HC�f���
@
=�'��&{B�#�                                    Bx[���  �          @p  ��ff?��H�(Q��3  C�Ϳ�ff@��
=�ffC :�                                    Bx[���  �          @r�\���@��&ff�,�CǮ���@ff��
�{C �f                                    Bx[��H  T          @l(���33@�\����"�HC���33@��
=�G�C.                                    Bx[���  T          @��H���R@  �,���'Ck����R@%��Q��z�B�\                                    Bx[��  �          @��
�  @�\�H���7��C	���  @��5�"\)C��                                    Bx[��:  �          @[���{?��{�!�C���{@���H��C�3                                    Bx[��  �          @U����
?���(��<p�C�3���
?��H��R�(ffC=q                                    Bx[��  �          @U���=q?������9z�CǮ��=q?ٙ�����%�C	�                                    Bx[�$,  �          @c33��\?������)�RC@ ��\?�Q��	���\)Cc�                                    Bx[�2�  �          @dz���?�33�ff�$�RC���?����ff��\C	J=                                    Bx[�Ax  �          @]p��  ?�����G����HC0��  @
�H��p�����C�                                    Bx[�P  �          @|�Ϳ�=q?�=q�Fff�P\)C@ ��=q?�p��7
=�:�C��                                    Bx[�^�  �          @��H��(�?��AG��>�C���(�@z��.{�'z�Cc�                                    Bx[�mj  T          @\)��\?�ff�-p��+z�C�{��\@
=q�����C��                                    Bx[�|  �          @u���?��H���(�C\)��@G�����C+�                                    Bx[���  T          @mp��z�?��\)�
=C���z�@�ÿ��H��ffC	ff                                    Bx[��\  �          @n{��\?��R������C���\@G������
CB�                                    Bx[��  �          @e���
=?�33��#G�Cn��
=@p��33��RC�                                    Bx[���  �          @g
=��z�@
�H����\)CaH��z�@�Ϳ�����RB��f                                    Bx[��N  �          @|���z�?��H�!G���RC�
�z�@33�{��CL�                                    Bx[���  �          @|���(�@
=q��R�=qCn�(�@\)����ffC\)                                    Bx[��  �          @z�H��
@   ����=qC+���
@������RCǮ                                    Bx[��@  �          @9����ff?�G����R��=qC�)��ff?��H��(��ģ�C��                                    Bx[���  �          @2�\��(�?�p������(�C����(�?�33��ff���C33                                    Bx[��  �          @4z�ٙ�?�G���33���CͿٙ�?�Q쿐������C5�                                    Bx[�2  �          @5�����?�녿�\)��  C
ff����?�=q��\)��  Cc�                                    Bx[�+�  �          @5����?�Q쿼(�����C�
����?�녿��R��C�R                                    Bx[�:~  �          @C33�	��?�{�����C���	��?��Ϳ�����p�Cu�                                    Bx[�I$  �          @?\)�ff?�Q��ff���\Cs3�ff?�zῨ����  C�                                    Bx[�W�  �          @@  ����?����
=�G�CǮ����?�\��
=��=qC	��                                    Bx[�fp  �          @>�R��
=?Ǯ��33�=qC
��
=?�������p�C	5�                                    Bx[�u  �          @8Q��\)?��ÿ��
� =qC�)��\)?�����
��(�CE                                    Bx[���  �          @Dz��(�?�(��˅���RC
�f��(�?�Q쿨����p�Cu�                                    Bx[��b  �          @A녿���?�녿����RC녿���?�\)��\)����C33                                    Bx[��  �          @AG���(�?�p���  ��HC\��(�?�p���G���C
��                                    Bx[���  �          @^{��p�?��R�G����C޸��p�@G���Q���  C�                                    Bx[��T  �          @_\)�z�?�ff�	����\C��z�@
=����� Q�C��                                    Bx[���  �          @Q녿���?�׿��H��C�q����@
=q��33��(�C
=                                    Bx[�۠  �          @W���p�?�׿��H�Q�C� ��p�@
=q�����C�                                    Bx[��F  �          @Y���\)?��
�   ��RC���\)?��ÿ޸R��z�C�                                    Bx[���  �          @S�
���?�\)� ���{C�����?�z��\�
=C�                                    Bx[��  �          @q��@Q��{��RCaH�@p���{���C\)                                    Bx[�8  T          @dz��Q�@=q�G��
�B�Q��Q�@,�Ϳ�\)�׮B�Q�                                    Bx[�$�  �          @j=q��{@���p��B����{@-p�������Q�B���                                    Bx[�3�  �          @~�R���@��  �	�RC����@0�׿�=q��p�C �\                                    Bx[�B*  �          @|(��
=q@Q������C8R�
=q@-p�������{C �\                                    Bx[�P�  �          @z�H���@�  ��C����@+�����(�C^�                                    Bx[�_v  �          @����
=@z��z��
=Cu��
=@*=q��z���{C�                                    Bx[�n  T          @����$z�@Q��p�� �RC	8R�$z�@-p��������C��                                    Bx[�|�  �          @���#33@Q��=q��C{�#33@\)����\)C�
                                    Bx[��h  �          @��\�#�
@
=q��ffC�)�#�
@!G�������  C��                                    Bx[��  �          @�  �1G�@  ����C�R�1G�@%�����G�C�                                    Bx[���  �          @���5@#�
��
��RC	��5@7���{��(�C��                                    Bx[��Z  �          @z=q�$z�@G�����
=C�f�$z�@���
�ظRC	�f                                    Bx[��   �          @tz��)��?�(������C���)��@33����RC@                                     Bx[�Ԧ  �          @�  �1G�?�\)�
�H�(�C��1G�@p���=q�ڣ�Ch�                                    Bx[��L  �          @�Q��,��@ ���
�H�ffC^��,��@��ff��=qC�                                    Bx[���  �          @��H�1G�@z��(����C.�1G�@=q����Q�C
�                                    Bx[� �  �          @|���)��@�
�z���33C&f�)��@Q��Q����C
{                                    Bx[�>  �          @vff�$z�@�\�G����HC�=�$z�@ff�����  C	�                                     Bx[��  �          @r�\���@
=��p����C=q���@�H�˅��{CaH                                    Bx[�,�  �          @q���\@�\�������RC�f��\@%���
��
=CaH                                    Bx[�;0  �          @o\)�:�H?�{��33���
C�R�:�H?��У���G�C=q                                    Bx[�I�  �          @w��G
=?�  ����Q�C+��G
=?Ǯ���ͅCaH                                    Bx[�X|  �          @s33�H��?��������
=C �f�H��?�녿У��˙�C
                                    Bx[�g"  �          @i���@  ?�=q��  ��=qC ��@  ?�\)���
���
Cff                                    Bx[�u�  �          @����P��?�(���=q���HC�\�P��?�G���ff���C�)                                    Bx[��n  �          @�  �i��?�=q��(����\C�3�i��?�{���H��=qC(�                                    Bx[��  T          @����j=q?��R��{��33C�{�j=q?�  ������=qCk�                                    Bx[���  �          @�Q��aG�?�(���
=����C�)�aG�?��R������p�Cn                                    Bx[��`  �          @�  �Y��?�\)��p�����C)�Y��@	����\)���HC��                                    Bx[��  �          @�  �S�
?�(�����ǮCQ��S�
@  ��z����C�                                    Bx[�ͬ  �          @����O\)@�
��33��33C�
�O\)@
=��  ���\C޸                                    Bx[��R  �          @���N{@녿�\)�хC���N{@���(����HC�                                    Bx[���  �          @��
�W
=?��R�����ԣ�C^��W
=@�
�Ǯ��C�                                     Bx[���  �          @����Z�H?��Ϳ��ʣ�C���Z�H@	����(���ffC��                                    Bx[�D  �          @����^�R?��ÿ�
=����CT{�^�R@ff������Q�C�H                                    Bx[��  �          @���E@�R������ffC8R�E@!G�������
=CǮ                                    Bx[�%�  �          @�33�3�
@'����ޣ�C���3�
@<�Ϳ�G���
=C�
                                    Bx[�46  �          @����4z�@$z��z��ԣ�C	���4z�@7���33���CxR                                    Bx[�B�  �          @�  �5�@'
=��ff���C	E�5�@9�����
��ffCJ=                                    Bx[�Q�  �          @�  �>{@ �׿�(����\C�q�>{@2�\��(����C�=                                    Bx[�`(  �          @�
=�@  @(���z���(�C�H�@  @-p���
=��(�C	�                                    Bx[�n�  �          @����R�\@
�H�����\)C���R�\@Q�aG��DQ�C�                                    Bx[�}t  �          @���Fff@�H��{��
=C
=�Fff@(Q�c�
�D��C�                                    Bx[��  �          @����C33@#�
���
��  C��C33@3�
���\�]�C	^�                                    Bx[���  �          @�(��B�\@.{�������RC
(��B�\@>{���
�Z{C��                                    Bx[��f  T          @��\�HQ�@$z��G���G�C���HQ�@3�
�}p��TQ�C
�                                    Bx[��  �          @����G�@*�H��=q��Q�Ck��G�@:�H����[33Cٚ                                    Bx[�Ʋ  �          @��R�C33@7
=��G���33C�=�C33@Fff�p���B�HC�                                    Bx[��X  �          @�Q��C33@(���Q�����Cff�C33@-p���Q�����C
Y�                                    Bx[���  T          @����:�H@�ÿ���G�C���:�H@'��p���V�\C
�                                    Bx[��  �          @w
=�@  @녿W
=�H(�C� �@  @���\��z�CW
                                    Bx[�J  T          @u��B�\@p��=p��2�\C��B�\@zᾙ�����RC�3                                    Bx[��  �          @o\)�=p�@��n{�f{C���=p�@{�   ��Q�C)                                    Bx[��  �          @~{�C�
@
=������p�Cc��C�
@�h���T��C��                                    Bx[�-<  �          @�Q��>�R@�
�����33CE�>�R@ �׿O\)�<Q�Cٚ                                    Bx[�;�  �          @|���C�
@ff�����33C���C�
@�
�Y���G�
C��                                    Bx[�J�  �          @xQ��?\)@\)����xz�C.�?\)@���z��z�C@                                     Bx[�Y.  T          @tz��>{@G��G��=p�C���>{@Q쾣�
����CL�                                    Bx[�g�  �          @z�H�G
=@{�0���"�HC}q�G
=@�
�k��Z=qCc�                                    Bx[�vz  �          @x���Dz�@(��s33�a�C���Dz�@����H��\)CǮ                                    Bx[��   T          @i���AG�?��+��)p�C���AG�@ �׾�=q����CL�                                    Bx[���  �          @s33�AG�@
=�fff�\z�C
=�AG�@  ����\)CO\                                    Bx[��l  
�          @u�A�@�ͿG��;�C��A�@�
���
��  C�f                                    Bx[��  T          @u��AG�?�
=�����z�Cc��AG�@
=q�c�
�W�
Cu�                                    Bx[���  �          @r�\�L(�?�녿��H��p�C���L(�?�{�Y���Pz�CǮ                                    Bx[��^  �          @qG��L(�?�\)�333�+�C���L(�?�(���������CL�                                    Bx[��  �          @q��B�\@(��Ǯ��C5��B�\@�R=�\)?��
C��                                    Bx[��  �          @w
=�G�@�Ϳ
=�(�C�
�G�@G�����C�                                    Bx[��P  �          @w
=�G
=@{��\���Ck��G
=@녽#�
�+�C��                                    Bx[��  �          @p  �G�@33���
���C�G�@z�=�G�?�Cz�                                    Bx[��  �          @j�H�HQ�?�33�u�s33C���HQ�?�\)>���@��C)                                    Bx[�&B  T          @_\)�;�?�{<#�
>�C���;�?���>���@љ�C�                                    Bx[�4�  �          @U�,(�?�Q�?(��A;�C�\�,(�?\?�  A�\)C�\                                    Bx[�C�  �          @\(��A�?���=���?ٙ�CB��A�?��>�(�@��C)                                    Bx[�R4  �          @U��Dz�?��
=�G�?��Cc��Dz�?�p�>\@ϮC5�                                    Bx[�`�  T          @Vff�K�?}p�>��R@��C"�)�K�?h��?�\A�C$\                                    Bx[�o�  �          @S�
�I��?Q�?�A�HC%T{�I��?0��?:�HAL��C'��                                    Bx[�~&  �          @S�
�HQ�?W
=?(��A8(�C$�3�HQ�?.{?Q�Af�RC'�3                                    Bx[���  �          @Dz��8��?aG�?\)A(��C#��8��?=p�?=p�A]C%�3                                    Bx[��r  �          @5�*=q?W
=?�A(��C"���*=q?5?0��A`  C%�                                    Bx[��  �          @1G��#�
?E�?.{AeC#J=�#�
?��?Tz�A���C&��                                    Bx[���  �          @2�\��?&ff?��A���C$����>�Q�?�33A�p�C+xR                                    Bx[��d  �          @'��(�?(�?�p�A���C$Y��(�>���?�=qA��C+T{                                    Bx[��
  �          @*�H��>�?��A�  C({��>k�?�Aי�C.!H                                    Bx[��  
�          @�R�	��>\)?�A�C0T{�	����?�A��C7(�                                    Bx[��V  �          @%����?xQ�?+�As\)C�����?L��?^�RA��RC h�                                    Bx[��  �          @(���p�>�Q�?��RA�C*�{�p�=��
?��A�C2                                      Bx[��  �          @�
�z�>�?:�HA��
C&���z�>��R?Q�A���C+z�                                    Bx[�H  �          @!G��z�?�?E�A�p�C'}q�z�>���?\(�A�C+�{                                    Bx[�-�  �          @�R��׾\)?�33A�p�C8@ ��׾Ǯ?��A��C?�q                                    Bx[�<�  �          @!���>�(�?(��Av�\C)����>�\)?=p�A��\C-T{                                    Bx[�K:  �          @.{� ��?\(�?�AB{C!�� ��?5?@  A�
C$)                                    Bx[�Y�  �          @*=q�
=?�  ?(��AeG�C�
=?Tz�?\(�A�  C �H                                    Bx[�h�  �          @(Q���
?���?
=qA=G�C}q��
?s33?E�A���C�=                                    Bx[�w,  �          @)���G�?�  ?   A.=qC!H�G�?�{?E�A�
=C��                                    Bx[���  �          @#�
���?�{>��A�\C� ���?�p�?B�\A�G�C�                                    Bx[��x  �          @.�R���R?޸R?   A,��C
�\���R?�=q?aG�A�G�C�                                     Bx[��  �          @1G���(�?�z�>L��@��
C����(�?�?�RAQ�C	Y�                                    Bx[���  �          @2�\��?�z�>aG�@���C�q��?�?&ffAV�\C
E                                    Bx[��j  �          @.{��33?�z�=u?��\C� ��33?���>��HA(��C�R                                    Bx[��  �          @/\)��{?��R��Q��p�C��{?��H>�Q�@���C�                                    Bx[�ݶ  �          @'���\)@�\�#�
���C ����\)?��R>�A ��C8R                                    Bx[��\  �          @<�Ϳ�=q@{>#�
@J=qC����=q@Q�?+�AUp�C                                    Bx[��  �          @H�ÿ��H@1녽#�
�5B�p����H@-p�?��A1�B�{                                    Bx[�	�  �          @G���@3�
?�\A�B�B���@'�?�33A��B�8R                                    Bx[�N  �          @C33��Q�@(�?s33A�z�B�8R��Q�?�z�?�Q�A�\)CxR                                    Bx[�&�  �          @Dz����?���?\(�A�p�C�����?���?��\A��HC��                                    Bx[�5�  �          @A��z�?�
=?z�HA���C+��z�?�33?��AҸRC޸                                    Bx[�D@  �          @@  �  ?���?�
=A��C��  ?�  ?�G�A�33C�                                    Bx[�R�  �          @(���  ?�ff?�Ai�CͿ�  ?�33?O\)A��C��                                    Bx[�a�  �          ?B�\��33?(��.{�`(�B�L;�33?!G����Q�B�ff                                    Bx[�p2  �          ?G���>�׽�Q����C	� ��>�<��
?�(�C	�                                    Bx[�~�  �          ?�녿8Q�?5�L���1�C^��8Q�?=p������CO\                                    Bx[��~  �          ?�z�>���.{?ǮB��=C�@ >����?�p�B��C���                                    Bx[��$  �          @(��=p�?�?�BffC^��=p�=���@   B��qC,33                                    Bx[���  T          @녾�ff?�?�33B�aHC xR��ff=���?�p�B��HC'�3                                    Bx[��p  �          ?��u<��
?�\)B�=qC.
=�u��Q�?�=qB�B�Cl�)                                    Bx[��  �          ?�=q��G����
?�ffB���C?^���G�����?�G�B��3C{u�                                    Bx[�ּ  �          ?Ǯ�W
=�.{?��
B��)CZ\)�W
=��?�Q�B��qCx+�                                    Bx[��b  �          ?�=q�\)��?��B���C\�=�\)��(�?�  B���C|�                                    Bx[��  �          ?��ü��u?��
B�Q�C��������H?p��Bz�
C�Z�                                    Bx[��  �          ?���>�����?�=qB�p�C��H>��#�
?��HBvC���                                    Bx[�T  �          @�
>�׿&ff?��B�C���>�׿��?�BZ(�C���                                    Bx[��  �          ?��H?+��#�
?��BFz�C�f?+��c�
?fffB�C�o\                                    Bx[�.�  �          ?��?aG������Q���33C���?aG��.{��
=��(�C��f                                    Bx[�=F  �          ?�ff?=p�=�\)����s�H@���?=p�>Ǯ����cp�A���                                    Bx[�K�  �          ?�33>\=���ff�)A��>\>�(���p�ǮBB=q                                    Bx[�Z�  
(          >�=���>u    >�  B��=���>u=uAV�\B���                                    Bx[�i8  �          >�G���<��
>W
=Bc��C*�f����>L��Ba�C@��                                    Bx[�w�  �          ?O\)>L��=L�ͽ��
��(�Ay>L��=�\)��\)��=qA�\)                                    Bx[���  �          ?�=q>u�\)?��HB���C�>u��(�?��B�(�C�Q�                                    Bx[��*  �          ?�\)�L�;�p�?�  B�C�  �L�ͿL��?���B~
=C��                                    Bx[���  �          @p�=��Ϳ���@
=Bw�C�y�=��Ϳ�\)?�G�B=ffC���                                    Bx[��v  �          @=q�0�׾��
@��B��qCL���0�׿^�R@
=Bx��Cg�                                    Bx[��  �          @���ff>�Q�?��Bn��C!0���ff��?�Bt��C;�                                    Bx[���  �          @#�
�h��?�\)@B]�
C!H�h��?��@�
B�p�C��                                    Bx[��h  �          @)������?���@�BK��B�k�����?��
@�B��HB���                                    Bx[��  �          @(�ÿ�\?��@{B���Cff��\�L��@#33B�L�C9p�                                    Bx[���  �          @(Q�W
=�8Q�@#33B�� C]  �W
=�Q�@�HB�B�C�f                                    Bx[�
Z  �          @(�=���{?�  BP=qC�p�=���G�?��Bp�C��                                    Bx[�   	�          @<(�?xQ�?�=q@\)BV�
BW��?xQ�?5@   B��3B                                      Bx[�'�  �          @N{?p��>u@$z�B���Ab�\?p�׾�(�@"�\B���C���                                    Bx[�6L  �          @S33?�(�����@>�RB�k�C��?�(���ff@333Bp��C���                                    Bx[�D�  �          @Q�?�ff?   @333Bo{A��?�ff�k�@5Bt�C���                                    Bx[�S�  �          @L(�?xQ�?Y��@8��B���B$�?xQ�=��
@@��B���@���                                    Bx[�b>  �          @Mp�>��?aG�@@  B��fB���>��=��
@HQ�B�=qA��                                    Bx[�p�  �          @J�H��p�?�@1�Bx�
B�녾�p�?&ff@C�
B��B�3                                    Bx[��  T          @AG��h��@�?�B��B���h��?��R@�BR=qB�                                    Bx[��0  �          @.{��z�?�G���\)���C �R��z�@�
�\(���ffB���                                    Bx[���  �          @,�Ϳ��H?�zῐ����B������H@�ÿ
=�W�B���                                    Bx[��|  �          @{��
=?���  ��ffB�k���
=?�>��@�
=B�u�                                    Bx[��"  �          @�R��?�녾k���B�����?��>�=q@�  B���                                    Bx[���  �          @0  ��(�@�������
B��H��(�@�
����>�\B��)                                    Bx[��n  T          @1G����@p���Q���z�B�\)���@�Ϳ\)�>�\B���                                    Bx[��  �          @9���Q�?�����(����C���Q�?�G����333C��                                    Bx[���  �          @5��?�33>k�@�ffC�R��?��?
=AB�HC�R                                    Bx[�`  �          @%��p�?���Q��	��C'���p�?
=q=L��?���C'��                                    Bx[�  �          @*=q�#�
?   ���
���C(�
�#�
?녾8Q��w
=C'c�                                    Bx[� �  �          @1��#�
?8Q쾽p����C$Q��#�
?J=q�#�
�]p�C"�
                                    Bx[�/R  �          @9���,(�>�Q�z�H���C,ff�,(�?(��Y�����C'B�                                    Bx[�=�  �          @0���'
=>L�ͿO\)���
C/���'
=>�녿:�H�vffC*�q                                    Bx[�L�  �          @3�
�,(�=�Q�G���p�C2(��,(�>�z�:�H�qG�C-Ǯ                                    Bx[�[D  �          @3�
�+�>k��G����C/�+�>�G��0���d  C*�=                                    Bx[�i�  �          @333�.�R>�{�����C,���.�R>�ff��Q���C*��                                    Bx[�x�  �          @1��-p�>��R�����HC-���-p�>�
=��p����HC+.                                    Bx[��6  �          @333�0  >�<��
>��HC*� �0  >�G�>��@B�\C*��                                    Bx[���  �          @9���6ff>�(�>W
=@�{C+ff�6ff>�Q�>��
@˅C,ٚ                                    Bx[���  �          @<(��7�>�ff=�G�@�C+0��7�>���>aG�@�C,
                                    Bx[��(  �          @:�H�.�R?}p��W
=���C &f�.�R?�G�=�\)?��C�                                    Bx[���  �          @:=q�(Q�?�Q�\)�333C�R�(Q�?�
=>B�\@o\)C�
                                    Bx[��t  �          @>�R�2�\?:�H?(�A?
=C%G��2�\?
=q?J=qAw�
C)
                                    Bx[��  �          @0���%?�?0��AiG�C(h��%>��
?O\)A�p�C,��                                    Bx[���  �          @0  ���>�G�?���Aљ�C)�R���=�Q�?��
A߅C1��                                    Bx[��f  �          @3�
���?�?�  B=qC%�����>\)?���B(�C0��                                    Bx[�  �          @7
=��?Y��?�ffB{CG���>��
?��HB.��C+
=                                    Bx[��  �          @C�
���?(��?�\)B�C$�R���>8Q�?޸RB�\C/                                    Bx[�(X  �          @:=q�*=q���
?���A��
C4\)�*=q���R?��
A�C:��                                    Bx[�6�  T          @C�
�4z�?O\)?W
=A�=qC#�q�4z�?
=q?��
A�z�C)�                                    Bx[�E�  �          @;���\?h��?���B��C�
��\>�p�@   B0
=C)��                                    Bx[�TJ  �          @@���(�?��
?��HB
=C�R�(�?�?�
=B"�C&                                    Bx[�b�  �          @P���  ?�p�?��
B
=C�=�  ?n{@Q�B$��C�
                                    Bx[�q�  �          @S33��ff?�=q@��B;�\C����ff?(�@*�HBZ=qC!W
                                    Bx[��<  �          @J�H��Q�?\@G�B=
=C�쿸Q�?Q�@&ffBe�
Ch�                                    Bx[���  �          @Mp�� ��?�p�?�p�A�  Cs3� ��?�ff?�p�B	G�CL�                                    Bx[���  �          @8Q쿓33?(�?\(�BG�C\��33>�33?�  BffC#Y�                                    Bx[��.  �          @QG��s33���@8��B�Cc���s33����@�RBI��Crk�                                    Bx[���  �          @Q녿xQ쿚�H@7
=Bv=qCgQ�xQ��(�@��B>=qCs�{                                    Bx[��z  �          @fff����(�@9��BYp�Ck𤿕��R@�\B�
Ct��                                    Bx[��   �          @c�
��\)�
=@&ffB=�HCr{��\)�0��?��B�
Cw�R                                    Bx[���  �          @W
=��
=����@#33BF\)Cm  ��
=�{?�
=B\)Ctn                                    Bx[��l  T          @U���p����@%BRG�C8@ ��p��aG�@(�BA�RCL�                                    Bx[�  �          @Y������{?��B	�Ci�=����*=q?�A��RCnk�                                    Bx[��  �          @^{�fff�C33?aG�Aw
=C}���fff�J�H���z�C~:�                                    Bx[�!^  �          @dz��ff�'�?�33A�  CoY���ff�:=q?
=A"�RCq�                                    Bx[�0  �          @j=q��\�!G�?�33A��Cj�׿�\�=p�?���A��Co�                                    Bx[�>�  �          @o\)���,��?�  A��\ChW
���AG�?(��A#�
Ckc�                                    Bx[�MP  �          @g���p��
=q@ffB�
Cc�\��p��+�?��A��Ci��                                    Bx[�[�  �          @q녿�z��6ff?�(�A��RCs����z��S33?��A�=qCv޸                                    Bx[�j�  �          @qG����R�+�@��B�RCp����R�L(�?��\A�
=Ct�                                    Bx[�yB  �          @z=q���,(�@G�B��Cn0����N�R?��A��Cr�R                                    Bx[���  �          @tzῪ=q���R@<��BK��Cl#׿�=q�0��@\)B��Ct33                                    Bx[���  �          @g������ ��@&ffB;�Ck�������+�?�33B
=CrǮ                                    Bx[��4  �          @Q녿333�#�
@H��B�L�CA8R�333��=q@=p�B�#�Cm�                                    Bx[���  T          @hQ����>�{@e�B�=qC5þ��Ϳ8Q�@a�B��RCp�R                                    Bx[�  �          @qG��(��>�\)@l��B��C�\�(�ÿQ�@g�B��3Cg�                                    Bx[��&  �          @o\)�J=q�#�
@h��B��=C70��J=q��{@^{B�{Cj��                                    Bx[���  �          @n�R����W
=@a�B��)C?B�������R@S�
B~�HCen                                    Bx[��r  �          @n�R����<�@aG�B�L�C2�H���׿�G�@W�B�u�C]Ǯ                                    Bx[��  �          @}p���  �\(�@eB��CV�쿠  ���@K�BZ�Cln                                    Bx[��  �          @����=q�u@p  B�8RC68R��=q��@dz�BtffCXu�                                    Bx[�d  T          @�녿�\)�:�H@s33Bx��CIW
��\)����@Z=qBR=qC`L�                                    Bx[�)
  �          @�����\��@e�BbQ�CA�R��\��{@P  BD��CW\                                    Bx[�7�  �          @���O\)>8Q�@Dz�B-��C0ٚ�O\)�=p�@>�RB'��C@�                                    Bx[�FV  �          @��H���
���
@��
B��fCgLͿ��
�!�@r�\BY(�Cw��                                    Bx[�T�  �          @�����Q쿐��@�{B��
CZ.��Q���@j=qBT��CnT{                                    Bx[�c�  �          @�  ��z�\@��\B�ǮCh� ��z��*�H@[�BF�
Cv�
                                    Bx[�rH  �          @���?^�R�&ff@uBY��C�C�?^�R�hQ�@8Q�Bp�C��                                     Bx[�  �          @�33����^�R@n{B]�CG���������@Q�B:�
C[�                                    Bx[�  �          @����33�7
=@7
=B'�
Cs�3��33�e�?�\)A�
=Cx�R                                    Bx[:  �          @�{���
�!�@>�RB=Cw�׿��
�S�
@�A�33C|�q                                    Bx[¬�  �          @���>�
=�/\)@C�
B?Q�C�Z�>�
=�a�@A�G�C�b�                                    Bx[»�  �          @�\)?zῨ��@{�B��C��{?z����@VffBT=qC���                                    Bx[��,  �          @�\)?0���6ff@VffBCQ�C��f?0���n�R@�A��\C�7
                                    Bx[���  �          @��Ϳ&ff�N{@7�B$�\C�O\�&ff�|(�?�G�A�  C�S3                                    Bx[��x  
�          @��ÿJ=q�j=q?��RA�(�C��f�J=q���H?J=qA*�RC��f                                    Bx[��  �          @�33���xQ�?��A�\)C�
����G����Ϳ�33C�>�                                    Bx[��  "          @}p���(��tzᾏ\)��33C����(��c33��
=���C���                                    Bx[�j  �          @�G��k��j=q?�33A��C�<)�k��\)>�ff@�\)C�aH                                    Bx[�"  
�          @�(��u�c33@A�=qC�|)�u����?h��AK\)C���                                    Bx[�0�  �          @��;����z�H?��\A���C��þ������
    =uC��3                                    Bx[�?\  �          @�{�Ǯ��G�?xQ�AW33C�Ff�Ǯ���;��
��33C�XR                                    Bx[�N  
�          @�33?0���n�R?���A��C�<)?0������>�{@�=qC���                                    Bx[�\�  �          @�z�>���xQ�?��A���C���>�����
>�?�  C���                                    Bx[�kN  "          @�{?0���|(�?���A�  C���?0�����
��\)�uC�                                    Bx[�y�  "          @��R?   ����?�  A[
=C�� ?   ������R��C�h�                                    Bx[È�  �          @��\����  ?+�Az�C��ͼ���Q�����
C���                                    Bx[×@  
�          @��?�\���\?�@���C���?�\��G��E��*=qC���                                    Bx[å�  "          @��
�#�
�tz�?��
A�=qC�˅�#�
��33>�=q@mp�C��                                     Bx[ô�  
�          @�p���z��l(�?�z�A��
C��f��z���33?(��A�RC���                                    Bx[��2  
Z          @�������fff?��HA�ffC��׾�������?@  A'\)C���                                    Bx[���  T          @���{�e�@�A�C�H���{���?c�
AEp�C���                                    Bx[��~  "          @���(���p��?�=qA���C���(����z�?\)@���C�o\                                    Bx[��$  �          @�p���(��hQ�@   A��C�����(���=q?B�\A(Q�C�                                    Bx[���  T          @���>�{��Q�?��A�(�C�t{>�{���=u?Q�C�Q�                                    Bx[�p  �          @�>aG����?�33A�ffC�� >aG���p�>�=q@b�\C�e                                    Bx[�  
�          @��R�\)�|��@33A���C��\)����?333A��C��                                    Bx[�)�  
�          @��Ϳ�\�k�@�B \)C���\��\)?�ffA^=qC��\                                    Bx[�8b  T          @��;�  �u�@	��A��C�)��  ���?Tz�A.�\C�Q�                                    Bx[�G  �          @��ͼ#�
�{�?�(�A�G�C��׼#�
��33?!G�A(�C��f                                    Bx[�U�  �          @�33>���u@   AۅC���>������?.{A��C��                                     Bx[�dT  T          @��þǮ�XQ�@�A��C���Ǯ�w�?p��AZ=qC�q                                    Bx[�r�  �          @\)�\�8Q�@*=qB)�C�@ �\�e�?�{A�z�C��R                                    Bx[ā�  �          @����P��@$z�B�HC��þ��y��?�33A��HC���                                    Bx[ĐF  "          @`  ��ff��?8Q�AծCK����ff�B�\?   A��CRh�                                    Bx[Ğ�  T          @y����ff@O\)�Tz��J�RB�.��ff@U�>u@h��B�                                    Bx[ĭ�  �          @k��@2�\��\)���C  �@.�R?�RAG�C��                                    Bx[ļ8  
Z          @h���@0  ��z���Ch��@,��?��A  C�                                    Bx[���  
�          @n{���@=p��������B�����@9��?(��A$��B�33                                    Bx[�ل  
�          @o\)�   @HQ���H��ffB�B��   @G�?
=qA��B�k�                                    Bx[��*  �          @h�ÿ�  @Tz�5�6{B�\)��  @W�>Ǯ@���B�                                    Bx[���  �          @q녿\@X�ÿ�R��RB�\)�\@Z=q?   @�{B��                                    Bx[�v  �          @tz�޸R@P�׿k��`  B�=q�޸R@XQ�>.{@(Q�B�z�                                    Bx[�  
�          @n�R��
@Fff<��
>�z�B�(���
@;�?�G�A}G�B�33                                    Bx[�"�  "          @g
=��@Dz�<#�
>B�\B�q��@9��?�  A�  B���                                    Bx[�1h            @j�H���
@I��?�A�
B����
@3�
?�G�A�p�B��q                                    Bx[�@  )          @k����@Vff>�
=@���B�����@A�?�p�A��B�B�                                    Bx[�N�  �          @qG����@c�
?
=qA�
BԳ3���@L(�?�z�A�Q�B�33                                    Bx[�]Z  �          @��R��z�@w
=�����B�uÿ�z�@l(�?��A{33B�\)                                    Bx[�l   �          @�33��33@u�����g
=B�.��33@mp�?�  AbffB�k�                                    Bx[�z�  �          @u���p�@`�׾��
��=qB��쿽p�@Z�H?W
=AJ{B��)                                    Bx[ŉL  "          @r�\�$z�@.{>���@�G�Cc��$z�@{?�A��RC
                                    Bx[ŗ�  �          @}p��1�@1G�>�ff@��C#��1�@{?�=qA�33C
c�                                    Bx[Ŧ�  �          @r�\�@��@	��?G�A?�Cc��@��?��
?�Q�A��C^�                                    Bx[ŵ>  
�          @_\)�8Q�?�\?E�AL��C}q�8Q�?�?�ffA��\CǮ                                    Bx[���  �          @l���K�?�G�?(�A��C
=�K�?��H?�33A��
CJ=                                    Bx[�Ҋ  �          @\)�^�R?W
=?�  A�{C&xR�^�R>���?�Q�A��HC/�                                    Bx[��0  �          @����^�R>�?��HA�C,k��^�R�W
=@ ��A�\)C7aH                                    Bx[���  T          @\)�a�?z�?޸RA��C*���a논��
?�=qA��C4h�                                    Bx[��|  T          @����c33>���?��
A�\)C-� �c33�L��?�A�C7=q                                    Bx[�"  T          @��H�j�H?�?��HA�Q�C+�R�j�H�u?��A�Q�C4�                                    Bx[��  �          @����c�
>�ff?�ffA�
=C,� �c�
�#�
?���A�G�C6�f                                    Bx[�*n            @����aG�>8Q�?�Q�A���C1��aG����?�33A��C;�{                                    Bx[�9  
�          @�G��^�R>#�
@G�A��HC1W
�^�R��\?��HA�
=C<ff                                    Bx[�G�  �          @����`  =���?��HA�{C2W
�`  ���?��A�(�C<�R                                    Bx[�V`  �          @�=q�\�ͽ�\)@��A�=qC5+��\�ͿE�@   A�\)C@��                                    Bx[�e  �          @����\�;�z�@z�A��RC8ٚ�\�Ϳu?�{A���CC�\                                    Bx[�s�  	�          @����]p��\(�?�A�p�CA�3�]p���z�?�A��HCJ.                                    Bx[ƂR  �          @\)�S�
�fff@ ��A�p�CC#��S�
��G�?���A�(�CL��                                    Bx[Ɛ�  �          @p���>�R��33?�Q�A��CI33�>�R��p�?�Q�A��CR0�                                    Bx[Ɵ�  �          @k��,(����R?��RB�HCP�f�,(���
?��A��CY}q                                    Bx[ƮD  
�          @l(��'
=��p�@	��BffCQ}q�'
=�
=?��
A�z�CZ��                                    Bx[Ƽ�  �          @n{�#33���@B
=CO�H�#33��\?�G�A�{CZ��                                    Bx[�ː  
�          @n{�+����
@33B��CQ�+��Q�?�A��HCZz�                                    Bx[��6            @vff�3�
��{@
=B�CI���3�
��?�{A�CU+�                                    Bx[���  [          @~�R�0�׿}p�@(Q�B'��CG�{�0�׿���@	��BG�CUc�                                    Bx[���  T          @~�R�   ��{@6ffB8z�CK���   �   @�
B33CZ��                                    Bx[�(  T          @����z`\)@3�
B8�
CR�=�z���R@(�B�HC_�H                                    Bx[��  
�          @\)�,(���
=@+�B)Q�CK���,(�� ��@Q�BG�CX                                    Bx[�#t  �          @q��Q쿦ff@'�B/��CP��Q��ff@�Bp�C]^�                                    Bx[�2  
�          @n�R�녿��@ ��B)ffCV����33?�A�p�Ca:�                                    Bx[�@�  �          @p������33@�HB!=qCWT{����?�(�Aۙ�Ca}q                                    Bx[�Of  �          @j=q��Ϳ�@  B33C[������\)?��RA�Cd��                                    Bx[�^  T          @N{��{��\@�
B7�
Ch^���{�(�?�=qA�\)Cp�
                                    Bx[�l�  �          @@  �s33���@�B4z�Cs!H�s33��R?�{A�ffCy                                    Bx[�{X  "          @HQ�Y���G�?���B(�Cyn�Y���2�\?���A��\C}�                                    Bx[ǉ�  "          @J=q��녿�z�@Q�B/  Co����� ��?�{A�p�Cu��                                    Bx[ǘ�  
�          @Dz��=q�Ǯ?�{BC\aH��=q�?�(�A��Cd�{                                    Bx[ǧJ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ǵ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�Ė              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�+               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�9�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�Hl              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�W              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�e�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�t^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ȃ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ȑ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ȠP              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[Ȯ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[Ƚ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��B              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�$&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�2�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�Ar              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�md              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�|
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[Ɋ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[əV              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ɧ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ɶ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��H              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�,              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�:x              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�I              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�W�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�fj              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�u              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ʃ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ʒ\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ʡ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ʯ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ʾN              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�ۚ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�$�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�3~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�B$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�P�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�_p              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�|�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ˋb              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[˚              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[˨�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[˷T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�Ԡ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[� �              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�;*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�I�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�Xv              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�g              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�u�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[̄h  @          @�Q����@8��B*��C]�����>�R?��HA��Cg�3                                    Bx[̓  	�          @��{�ff@Mp�B9��C_p��{�Fff@  A�\Cjz�                                    Bx[̡�  T          @�(���p��z�@G�B6��Ces3��p��QG�@A��Cn��                                    Bx[̰Z  
�          @�����z���@Dz�B6{Cf�=��z��QG�@�\A�ffCo��                                    Bx[̿   "          @��������@U�BHffCj�q�����Tz�@33A��
CtG�                                    Bx[�ͦ  T          @�G����ÿ��R@\(�BV��Cg�q�����E@ ��B�Cs�                                    Bx[��L  T          @��Ϳ�(���{@`��Bh�RCc�=��(��0  @-p�B#��Cq��                                    Bx[���  �          @����G���(�@L(�BIQ�Cd5ÿ�G��>�R@�BQ�Coc�                                    Bx[���  �          @���ff��
=@>{B7\)CW��ff�(Q�@�A���CdO\                                    Bx[�>  �          @��
��H��{@O\)B@z�CU�H��H�)��@��B	G�Cc�3                                    Bx[��  �          @��H��ÿ�\@Tz�BH=qC[������5@{B
=Ci
=                                    Bx[�%�  
�          @�33�B�\���R@-p�B33CJ=q�B�\�Q�@A��CW                                      Bx[�40  
�          @���r�\�:�H?��AЏ\C>��r�\��{?�  A��
CG�3                                    Bx[�B�  	�          @����s33��\?�33A�Q�C;���s33��33?˅A�=qCD��                                    Bx[�Q|  �          @����o\)��z�@33A�p�C8k��o\)��  ?���A���CC                                      Bx[�`"  �          @���z�H�B�\?�  A��HC6� �z�H�L��?���A��C?�\                                    Bx[�n�  �          @��\��=q�#�
?��HA�C4{��=q��?�{A��RC;\)                                    Bx[�}n  
�          @�����  =#�
?�=qA�(�C3aH��  ��?��RA�{C;s3                                    Bx[͌  �          @���~{���
?�z�A�G�C4E�~{�(�?��A�  C<�R                                    Bx[͚�  N          @�(����\=L��?���A���C3@ ���\��\?��RA�33C;#�                                    Bx[ͩ`  
.          @��H��(�����?�Q�A���C5p���(��J=q?�\A��HC>�\                                    Bx[͸  T          @��R���þ#�
?���A��C6+����ÿY��?�G�A�(�C?33                                    Bx[�Ƭ  
�          @�z�����>aG�?��
A��C1���׾�(�?�  A�
=C9��                                    Bx[��R  
�          @����\>�z�?�G�A�ffC0.���\��Q�?޸RA�
=C8��                                    Bx[���  
�          @������R>#�
?�(�A�p�C1�
���R��?�z�A�G�C:L�                                    Bx[��  
�          @������=�Q�?�(�A�p�C2�{���׿�?��A���C;)                                    Bx[�D  (          @�����p�<#�
?��
A���C3����p���R?�A�  C<                                    Bx[��  
(          @�{����<�?�\)A�  C3�)���Ϳ��?��
A�  C;&f                                    Bx[��  "          @�����33>k�?�{A�Q�C0�q��33��Q�?�=qA��C8�                                     Bx[�-6  T          @����
=>�G�?�{A�=qC.W
��
=��  ?�33A��
C78R                                    Bx[�;�  
�          @����Q�>�{?�A��\C/޸��Q쾏\)?�Q�A�C7Q�                                    Bx[�J�  T          @�33���?z�?�Q�A�=qC,}q������@G�A�z�C5�                                    Bx[�Y(  "          @�����{?
=q?޸RA���C-)��{��?�A���C5�                                    Bx[�g�  
�          @�=q��Q�?�?���A��HC-k���Q콣�
?�
=A�
=C5\                                    Bx[�vt  
�          @��\��  ?\)?�z�A���C,���  ��\)?�  A�C4�f                                    Bx[΅  
�          @�����R?J=q?��A�33C)�3���R=u@33A�G�C3.                                    Bx[Γ�  Z          @�p���ff?Tz�?�(�A���C)�R��ff=�\)@��A�C3)                                    Bx[΢f  "          @����{?J=q?��RA���C*n��{<�@��A�=qC3�f                                    Bx[α  �          @�(���>�G�@33A��\C.�������
@z�A��RC7��                                    Bx[ο�  T          @������?.{@ffA��HC+���녽�G�@��A��HC5Q�                                    Bx[��X  �          @��H��=q?#�
@�RA�33C,u���=q�L��@�
A���C6aH                                    Bx[���  "          @��
��=q?��@33Aʣ�C-xR��=q����@A���C7��                                    Bx[��  �          @������>�=q@p�Aљ�C0�)����
=q@
=qA�ffC:�)                                    Bx[��J            @������>�Q�@G�A�33C/}q������@  A�G�C9��                                    Bx[��  Z          @�G���p�?�@�AڸRC-���p���p�@p�Aݙ�C8�=                                    Bx[��  �          @����{?+�@��A�G�C+�f��{�k�@�RA�
=C6ٚ                                    Bx[�&<  
�          @�=q���
?fff@\)Aޏ\C(����
�#�
@(��A�(�C4�                                    Bx[�4�  "          @����?��@\)A�
=C&�
��>�@-p�A��HC2Y�                                    Bx[�C�  "          @�p�����?��R@\)Aڏ\C%{����>�=q@0��A��C0��                                    Bx[�R.  �          @�����(�?��\@p�A�33C$����(�>���@0  A�33C0=q                                    Bx[�`�  
�          @��
��z�?���@(�A�C%}q��z�>�  @,��A�C0�H                                    Bx[�oz  "          @�z����?��R@=qA�(�C%\���>���@,(�A�C0B�                                    Bx[�~   
Z          @����z�?���@#33A��HC&���z�=�@1G�A��RC2��                                    Bx[ό�  
�          @�{���?��@%�A��HC&�����=���@333A�(�C2                                    Bx[ϛl  "          @�(����\?�33@%A�Q�C%�����\>��@5�A��
C2)                                    Bx[Ϫ  
Z          @�33���?��@#�
A�Q�C%�q���>��@2�\A��
C2�                                    Bx[ϸ�  "          @������?��
@p�A�G�C$@ ����>��
@0��A�Q�C0�                                    Bx[��^  "          @�����\?8Q�@�HA���C*�����\�B�\@!G�A�G�C6xR                                    Bx[��  
�          @������\?(�@{A�{C+�����\���R@!�A��C8{                                    Bx[��  �          @�33��p�?.{@(�A���C+@ ��p��k�@!G�A�G�C6��                                    Bx[��P  
�          @��
���H?��@$z�A���C&�3���H=���@2�\A�=qC2�q                                    Bx[��  
�          @��H��=q?�Q�@�RA�\)C%s3��=q>k�@/\)A���C1!H                                    Bx[��  �          @����Q�?���@�HA�33C#���Q�>�(�@0  A�  C.�{                                    Bx[�B  N          @�����?���@�A��C ޸���?@  @!G�A�\C*��                                    Bx[�-�  Z          @�=q����?�=q@�A�
=C �q����?+�@,(�A�p�C+�\                                    Bx[�<�  "          @������R?ٙ�@p�A���C+����R?L��@*�HA�{C)�\                                    Bx[�K4  "          @�\)���R?�G�@A��\Ck����R?h��@%�A�\C(��                                    Bx[�Y�  �          @������?�(�@ ��A���C5�����?c�
@\)A�G�C(�)                                    